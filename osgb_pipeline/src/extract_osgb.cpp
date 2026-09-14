#include <osg/Geometry>
#include <osg/Geode>
#include <osg/PagedLOD>
#include <osg/Transform>
#include <osg/TriangleIndexFunctor>
#include <osgDB/ReadFile>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <functional>
namespace fs=std::filesystem;
struct Triangles { std::function<void(unsigned,unsigned,unsigned)> emit;
 void operator()(unsigned a,unsigned b,unsigned c){emit(a,b,c);} };
struct Extractor {
 fs::path out; double density; size_t limit, meshes=0, faces=0, points=0, files=0, transforms=0;
 unsigned long long offset=0;
 osg::Vec3d origin;
 std::ofstream obj,bin,manifest;
 std::set<fs::path> active;
 std::mt19937_64 rng{42}; std::uniform_real_distribution<double> uniform{0,1};
 Extractor(fs::path o,double d,size_t l,osg::Vec3d shift):out(o),density(d),limit(l),origin(shift){
  fs::create_directories(out); obj.open(out/"model.obj"); bin.open(out/"points.bin",std::ios::binary);
  manifest.open(out/"geometry.tsv"); obj<<std::setprecision(17); manifest<<"source\tvertices\ttriangles\n";
  if(!obj||!bin||!manifest) throw std::runtime_error("Cannot create output");
 }
 void file(const fs::path& p,osg::Matrixd m){
  if(limit && meshes>=limit)return;
  auto path=fs::canonical(p);
  if(!active.insert(path).second)throw std::runtime_error("Cyclic reference: "+path.string());
  osg::ref_ptr<osg::Node> node=osgDB::readNodeFile(path.string());
  if(!node)throw std::runtime_error("Cannot load: "+path.string());
  ++files; walk(node,m,path); active.erase(path);
 }
 void walk(osg::Node* n,osg::Matrixd m,const fs::path& source){
  if(limit && meshes>=limit)return;
  if(auto t=dynamic_cast<osg::Transform*>(n)){t->computeLocalToWorldMatrix(m,nullptr); ++transforms;}
  if(auto p=dynamic_cast<osg::PagedLOD*>(n)){
   // Only a single external refinement is accepted here. Spatial partitions
   // are Group children; ambiguous multiple external LOD alternatives fail closed.
   std::vector<std::string> refs;
   for(unsigned i=0;i<p->getNumFileNames();++i)if(!p->getFileName(i).empty())refs.push_back(p->getFileName(i));
   if(refs.size()>1)throw std::runtime_error("Multiple external LOD alternatives require inspection: "+source.string());
   if(!refs.empty()){
    fs::path ref=refs[0], resolved;
    for(auto candidate:{source.parent_path()/ref,fs::path(p->getDatabasePath())/ref,source.parent_path()/p->getDatabasePath()/ref})
     if(fs::is_regular_file(candidate)){resolved=candidate;break;}
    if(resolved.empty())throw std::runtime_error("Missing external reference: "+ref.string());
    file(resolved,m);return;
   }
  } else if(dynamic_cast<osg::LOD*>(n))throw std::runtime_error("Resident LOD requires explicit selection: "+source.string());
  if(auto geode=dynamic_cast<osg::Geode*>(n))for(unsigned i=0;i<geode->getNumDrawables();++i){
   if(limit && meshes>=limit)break;
   auto g=geode->getDrawable(i)->asGeometry();if(!g)throw std::runtime_error("Non-Geometry drawable");
   geometry(g,m,source);
  }
  if(auto group=n->asGroup())for(unsigned i=0;i<group->getNumChildren();++i)walk(group->getChild(i),m,source);
 }
 void geometry(osg::Geometry* g,const osg::Matrixd& m,const fs::path& source){
  auto v=dynamic_cast<osg::Vec3Array*>(g->getVertexArray());
  auto vd=dynamic_cast<osg::Vec3dArray*>(g->getVertexArray());
  if(!v&&!vd)throw std::runtime_error("Unsupported vertex array");
  std::vector<osg::Vec3d> verts;
  unsigned count=v?v->size():vd->size();
  obj<<"o geometry_"<<meshes<<"\n";
  for(unsigned i=0;i<count;++i){auto q=(v?osg::Vec3d((*v)[i]):(*vd)[i])*m+origin;
   for(int k=0;k<3;++k)if(!std::isfinite(q[k]))throw std::runtime_error("Nonfinite vertex");
   verts.push_back(q);obj<<"v "<<q.x()<<" "<<q.y()<<" "<<q.z()<<"\n";
  }
  size_t start=faces;
  osg::TriangleIndexFunctor<Triangles> fun;
  fun.emit=[&](unsigned a,unsigned b,unsigned c){
   auto A=verts.at(a),B=verts.at(b),C=verts.at(c);
   obj<<"f "<<offset+a+1<<" "<<offset+b+1<<" "<<offset+c+1<<"\n"; ++faces;
   double expectation=((B-A)^(C-A)).length()*0.5*density;
   if(!std::isfinite(expectation)||expectation>1e9)throw std::runtime_error("Invalid triangle sampling count");
   size_t n=static_cast<size_t>(std::floor(expectation));if(uniform(rng)<expectation-n)++n;
   for(size_t i=0;i<n;++i){double s=std::sqrt(uniform(rng)),t=uniform(rng);auto q=A*(1-s)+B*(s*(1-t))+C*(s*t);
    double xyz[3]={q.x(),q.y(),q.z()};bin.write(reinterpret_cast<char*>(xyz),sizeof(xyz));++points;}
  };
  g->accept(fun);offset+=count;++meshes;
  manifest<<source.string()<<'\t'<<count<<'\t'<<faces-start<<'\n';
  if(meshes%100==0)std::cerr<<"geometries="<<meshes<<" points="<<points<<std::endl;
 }
 void finish(){
  obj.flush();bin.flush();manifest.flush();if(!obj||!bin||!manifest)throw std::runtime_error("Output write failed");
  std::ofstream stats(out/"stats.json");stats<<"{\"files\":"<<files<<",\"geometries\":"<<meshes<<",\"triangles\":"<<faces<<",\"points\":"<<points<<",\"transforms\":"<<transforms<<"}\n";
 }
};
int main(int argc,char** argv){try{
 if(argc!=9){std::cerr<<"Usage: extract_osgb ROOT OUTPUT DENSITY MAX_GEOMETRIES ORIGIN_X ORIGIN_Y ORIGIN_Z SEED\n";return 2;}
 double d=std::stod(argv[3]);if(!std::isfinite(d)||d<=0)throw std::runtime_error("Invalid density");
 fs::path out=argv[2];if(fs::exists(out))throw std::runtime_error("Output must be a new directory");
 Extractor e(out,d,std::stoull(argv[4]),osg::Vec3d(std::stod(argv[5]),std::stod(argv[6]),std::stod(argv[7])));
 e.rng.seed(std::stoull(argv[8]));e.file(argv[1],osg::Matrixd::identity());e.finish();
 if(!e.faces||!e.points)throw std::runtime_error("Empty extraction");return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
