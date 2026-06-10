## Build
```
mkdir -p rasterman_ws/src
cd rasterman_ws/src
git clone https://github.com/gandres42/rasterman.git
git clone https://github.com/npragin/collective-construction.git
cd ..
colcon build
```

## Running
The three test options are included as launch files, accessible with:
```
ros2 launch rasterman dot.xml
ros2 launch rasterman line.xml
ros2 launch rasterman bill.xml
```
To run custom images, place your image in `src/rasterman` and run the node directly:
```
ros2 run rasterman rasterman --ros-args -p image:=custom.jpg
```

## Topics
`/structure_plan`: StructurePlan including block placements and types
`/rasterman/poses`: PoseArray of block centroids, used for visualization/debugging

