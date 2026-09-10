#include <osg/Geometry>
#include <osg/Geode>
#include <osg/PagedLOD>
#include <osg/Transform>
#include <osg/TriangleIndexFunctor>
#include <osgDB/ReadFile>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>

namespace fs = std::filesystem;

struct Triangles {
    std::function<void(unsigned, unsigned, unsigned)> emit;
    void operator()(unsigned a, unsigned b, unsigned c) { emit(a, b, c); }
};

// Minimal LAS 1.4 point-format-0 writer.  The output has XYZ only; the
// remaining point attributes are deliberately zero because OSGB has no
// original lidar intensity/return/classification values.
struct LasWriter {
    static constexpr uint16_t header_size = 375;
    static constexpr uint16_t record_length = 20;
    static constexpr double scale = 0.001;

    std::ofstream file;
    std::array<double, 3> offset{};
    std::array<double, 3> minimum{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    std::array<double, 3> maximum{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    uint64_t points = 0;
    std::vector<uint8_t> buffer;

    explicit LasWriter(const fs::path& path, std::array<double, 3> point_offset)
        : offset(point_offset), buffer() {
        if (fs::exists(path)) {
            throw std::runtime_error("LAS output already exists: " + path.string());
        }
        file.open(path, std::ios::binary | std::ios::trunc);
        if (!file) throw std::runtime_error("Cannot create LAS output: " + path.string());
        buffer.reserve(4 * 1024 * 1024);
        write_header();
        file.seekp(header_size, std::ios::beg);
    }

    template <typename T>
    void put(const T& value) {
        file.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    void fixed(const std::string& value, size_t length) {
        std::vector<char> bytes(length, '\0');
        std::memcpy(bytes.data(), value.data(), std::min(length, value.size()));
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    void write_header() {
        file.seekp(0, std::ios::beg);
        const char signature[4] = {'L', 'A', 'S', 'F'};
        file.write(signature, 4);
        put(uint16_t(0)); // File Source ID
        put(uint16_t(0)); // Global Encoding
        put(uint32_t(0)); // Project ID 1
        put(uint16_t(0)); // Project ID 2
        put(uint16_t(0)); // Project ID 3
        const uint8_t project_id_4[8] = {};
        file.write(reinterpret_cast<const char*>(project_id_4), 8);
        put(uint8_t(1));
        put(uint8_t(4));
        fixed("OpenSceneGraph", 32);
        fixed("osgb_pipeline", 32);
        put(uint16_t(0)); // File creation day
        put(uint16_t(2026));
        put(header_size);
        put(uint32_t(header_size)); // no VLRs
        put(uint32_t(0));
        put(uint8_t(0)); // point data record format 0
        put(record_length);
        const uint32_t legacy_count = points > UINT32_MAX
                                           ? UINT32_MAX
                                           : static_cast<uint32_t>(points);
        put(legacy_count);
        put(legacy_count);
        for (int i = 1; i < 5; ++i) put(uint32_t(0));
        put(scale);
        put(scale);
        put(scale);
        put(offset[0]);
        put(offset[1]);
        put(offset[2]);
        for (int i = 0; i < 3; ++i) {
            put(points ? maximum[i] : 0.0);
            put(points ? minimum[i] : 0.0);
        }
        put(uint64_t(0)); // Start of Waveform Data Packet Record
        put(uint64_t(0)); // Start of First EVLR
        put(uint32_t(0)); // Number of EVLRs
        put(points);      // Extended point count
        put(points);      // Extended return 1
        for (int i = 1; i < 15; ++i) put(uint64_t(0));
        if (file.tellp() != static_cast<std::streampos>(header_size)) {
            throw std::runtime_error("Internal LAS header size mismatch");
        }
    }

    int32_t quantize(double value, int axis) const {
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite sampled point");
        const double scaled = (value - offset[axis]) / scale;
        const long long integer = std::llround(scaled);
        if (integer < std::numeric_limits<int32_t>::min() ||
            integer > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("Coordinate exceeds LAS int32 range; choose another offset");
        }
        return static_cast<int32_t>(integer);
    }

    void flush_buffer() {
        if (buffer.empty()) return;
        file.write(reinterpret_cast<const char*>(buffer.data()),
                   static_cast<std::streamsize>(buffer.size()));
        buffer.clear();
        if (!file) throw std::runtime_error("LAS point write failed");
    }

    void point(const osg::Vec3d& value) {
        const int32_t x = quantize(value.x(), 0);
        const int32_t y = quantize(value.y(), 1);
        const int32_t z = quantize(value.z(), 2);
        const int32_t coordinates[3] = {x, y, z};
        const size_t start = buffer.size();
        buffer.resize(start + record_length, 0);
        std::memcpy(buffer.data() + start, coordinates, sizeof(coordinates));
        ++points;
        for (int i = 0; i < 3; ++i) {
            const double actual = offset[i] + static_cast<double>(coordinates[i]) * scale;
            minimum[i] = std::min(minimum[i], actual);
            maximum[i] = std::max(maximum[i], actual);
        }
        if (buffer.size() >= buffer.capacity() - record_length) flush_buffer();
    }

    void finish() {
        flush_buffer();
        write_header();
        file.flush();
        if (!file) throw std::runtime_error("LAS output flush failed");
        file.close();
    }
};

struct Extractor {
    double density;
    size_t limit;
    size_t geometries = 0;
    uint64_t faces = 0;
    uint64_t files = 0;
    uint64_t transforms = 0;
    osg::Vec3d origin;
    LasWriter writer;
    std::set<fs::path> active;
    std::mt19937_64 rng{42};
    std::uniform_real_distribution<double> uniform{0, 1};

    Extractor(fs::path output, double d, size_t l, osg::Vec3d shift,
              std::array<double, 3> point_offset)
        : density(d), limit(l), origin(shift), writer(output, point_offset) {}

    void file_node(const fs::path& path, osg::Matrixd matrix) {
        if (limit && geometries >= limit) return;
        const auto canonical = fs::canonical(path);
        if (!active.insert(canonical).second) {
            throw std::runtime_error("Cyclic reference: " + canonical.string());
        }
        osg::ref_ptr<osg::Node> node = osgDB::readNodeFile(canonical.string());
        if (!node) throw std::runtime_error("Cannot load: " + canonical.string());
        ++files;
        walk(node, matrix, canonical);
        active.erase(canonical);
    }

    void walk(osg::Node* node, osg::Matrixd matrix, const fs::path& source) {
        if (limit && geometries >= limit) return;
        if (auto transform = dynamic_cast<osg::Transform*>(node)) {
            transform->computeLocalToWorldMatrix(matrix, nullptr);
            ++transforms;
        }
        if (auto paged = dynamic_cast<osg::PagedLOD*>(node)) {
            std::vector<std::string> refs;
            for (unsigned i = 0; i < paged->getNumFileNames(); ++i) {
                if (!paged->getFileName(i).empty()) refs.push_back(paged->getFileName(i));
            }
            if (refs.size() > 1) {
                throw std::runtime_error("Multiple external LOD alternatives require inspection: " +
                                         source.string());
            }
            if (!refs.empty()) {
                fs::path ref = refs[0];
                fs::path resolved;
                for (const auto& candidate : {source.parent_path() / ref,
                                              fs::path(paged->getDatabasePath()) / ref,
                                              source.parent_path() / paged->getDatabasePath() / ref}) {
                    if (fs::is_regular_file(candidate)) {
                        resolved = candidate;
                        break;
                    }
                }
                if (resolved.empty()) {
                    throw std::runtime_error("Missing external reference: " + ref.string());
                }
                file_node(resolved, matrix);
                return;
            }
        } else if (dynamic_cast<osg::LOD*>(node)) {
            throw std::runtime_error("Resident LOD requires explicit selection: " + source.string());
        }
        if (auto geode = dynamic_cast<osg::Geode*>(node)) {
            for (unsigned i = 0; i < geode->getNumDrawables(); ++i) {
                if (limit && geometries >= limit) break;
                auto geometry = geode->getDrawable(i)->asGeometry();
                if (!geometry) throw std::runtime_error("Non-Geometry drawable");
                emit_geometry(geometry, matrix, source);
            }
        }
        if (auto group = node->asGroup()) {
            for (unsigned i = 0; i < group->getNumChildren(); ++i) {
                walk(group->getChild(i), matrix, source);
            }
        }
    }

    void emit_geometry(osg::Geometry* geometry, const osg::Matrixd& matrix,
                       const fs::path& source) {
        auto vertices = dynamic_cast<osg::Vec3Array*>(geometry->getVertexArray());
        auto vertices_d = dynamic_cast<osg::Vec3dArray*>(geometry->getVertexArray());
        if (!vertices && !vertices_d) throw std::runtime_error("Unsupported vertex array");
        const unsigned count = vertices ? vertices->size() : vertices_d->size();
        std::vector<osg::Vec3d> transformed;
        transformed.reserve(count);
        for (unsigned i = 0; i < count; ++i) {
            const osg::Vec3d local = vertices ? osg::Vec3d((*vertices)[i]) : (*vertices_d)[i];
            const osg::Vec3d value = local * matrix + origin;
            for (int axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(value[axis])) throw std::runtime_error("Nonfinite vertex");
            }
            transformed.push_back(value);
        }
        osg::TriangleIndexFunctor<Triangles> functor;
        functor.emit = [&](unsigned a, unsigned b, unsigned c) {
            const auto& a_point = transformed.at(a);
            const auto& b_point = transformed.at(b);
            const auto& c_point = transformed.at(c);
            ++faces;
            const double expected = ((b_point - a_point) ^ (c_point - a_point)).length() *
                                    0.5 * density;
            if (!std::isfinite(expected) || expected > 1e9) {
                throw std::runtime_error("Invalid triangle sampling count");
            }
            const size_t whole = static_cast<size_t>(std::floor(expected));
            size_t samples = whole;
            if (uniform(rng) < expected - whole) ++samples;
            for (size_t i = 0; i < samples; ++i) {
                const double s = std::sqrt(uniform(rng));
                const double t = uniform(rng);
                writer.point(a_point * (1 - s) + b_point * (s * (1 - t)) + c_point * (s * t));
            }
        };
        geometry->accept(functor);
        ++geometries;
        if (geometries % 100 == 0) {
            std::cerr << "geometries=" << geometries << " points=" << writer.points << '\n';
        }
    }

    void finish() {
        writer.finish();
        std::cerr << "files=" << files << " geometries=" << geometries
                  << " triangles=" << faces << " points=" << writer.points
                  << " transforms=" << transforms << '\n';
    }
};

int main(int argc, char** argv) {
    try {
        if (argc != 12) {
            std::cerr << "Usage: extract_las ROOT LAS_PART DENSITY MAX_GEOMETRIES "
                         "ORIGIN_X ORIGIN_Y ORIGIN_Z SEED OFFSET_X OFFSET_Y OFFSET_Z\n";
            return 2;
        }
        const double density = std::stod(argv[3]);
        if (!std::isfinite(density) || density <= 0) throw std::runtime_error("Invalid density");
        const fs::path output = argv[2];
        if (fs::exists(output)) throw std::runtime_error("LAS part already exists: " + output.string());
        const std::array<double, 3> point_offset{
            std::stod(argv[9]), std::stod(argv[10]), std::stod(argv[11])};
        Extractor extractor(output, density, std::stoull(argv[4]),
                             osg::Vec3d(std::stod(argv[5]), std::stod(argv[6]), std::stod(argv[7])),
                             point_offset);
        extractor.rng.seed(std::stoull(argv[8]));
        extractor.file_node(argv[1], osg::Matrixd::identity());
        if (!extractor.faces) throw std::runtime_error("Empty extraction");
        extractor.finish();
        if (!extractor.writer.points) throw std::runtime_error("Empty point extraction");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
