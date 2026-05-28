## Build and Run
```
mkdir -p rasterman_ws/src
cd rasterman_ws/src
git clone https://github.com/gandres42/rasterman.git
git clone https://github.com/npragin/collective-construction.git
cd ..
colcon build
ros2 run rasterman main
```
## Topics
`/rasterman/structure_plan`: StructurePlan including block placements and types
`/rasterman/viz`: Image showing image being built and centroids
`/rasterman/poses`: PoseArray of block centroids, used for visualization/debugging