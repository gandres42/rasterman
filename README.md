```
mkdir -p rasterman_ws/src
cd rasterman_ws/src
git clone https://github.com/gandres42/rasterman.git
git clone https://github.com/npragin/collective-construction.git
cd ..
colcon build
ros2 run rasterman main
```
