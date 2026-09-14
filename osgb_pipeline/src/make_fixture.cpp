#include <osg/Geometry>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/PagedLOD>
#include <osgDB/WriteFile>
#include <filesystem>
int main(int argc,char**argv){
 if(argc!=2)return 2;std::filesystem::create_directories(argv[1]);
 osg::ref_ptr<osg::Geometry> g=new osg::Geometry;
 osg::ref_ptr<osg::Vec3Array> v=new osg::Vec3Array;
 v->push_back({0,0,0});v->push_back({1,0,0});v->push_back({0,1,0});g->setVertexArray(v);
 g->addPrimitiveSet(new osg::DrawArrays(GL_TRIANGLES,0,3));
 osg::ref_ptr<osg::Geode> leaf=new osg::Geode;leaf->addDrawable(g);
 osg::ref_ptr<osg::MatrixTransform> child=new osg::MatrixTransform;
 child->setMatrix(osg::Matrixd::scale(2,2,2));child->addChild(leaf);
 auto dir=std::filesystem::path(argv[1]);osgDB::writeNodeFile(*child,(dir/"leaf.osgb").string());
 osg::ref_ptr<osg::PagedLOD> paged=new osg::PagedLOD;
 paged->setRangeMode(osg::LOD::PIXEL_SIZE_ON_SCREEN);
 paged->addChild(leaf,0,10);paged->setFileName(1,"leaf.osgb");paged->setRange(1,10,1e9);
 osg::ref_ptr<osg::MatrixTransform> root=new osg::MatrixTransform;
 root->setMatrix(osg::Matrixd::translate(10,20,30));root->addChild(paged);
 return osgDB::writeNodeFile(*root,(dir/"Block.osgb").string())?0:1;
}
