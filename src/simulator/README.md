# 仿真器

完整启动、路线和模型验证见[仿真 README](../../docs/simulation/README.md)。本包提供地图、目标运动、机器人运动学和三维传感器。

## 单独启动

先从项目根目录加载 ROS 和工作空间：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch simulator simulation.launch.py environment:=forest
ros2 launch simulator simulation_headless.launch.py environment:=offroad_mixed
ros2 launch simulator simulation.launch.py \
  map.file:="$TRACKER_ROOT/src/simulator/map/maze2.json"
```

| 配置 | 用途 |
| --- | --- |
| `config/simulator.yaml` | 默认演示参数 |
| `config/expert_collection.yaml` | 20×20 m 专家采集配置 |
| `config/environments/forest.yaml` | 稀疏森林 |
| `config/environments/forest_dense.yaml` | 高密度森林 |
| `config/environments/rocky.yaml` | 岩石障碍 |
| `config/environments/offroad_mixed.yaml` | 混合场景 |

`config:=FILE` 选择 YAML，`environment:=NAME` 选择预设；显式 YAML 优先。可覆盖 `map.seed`、`map.style`、`map.obstacles`、`map.width`、`map.height`。提供 `map.file` 后，地图尺寸及 tracker / target 初始位姿从 JSON 读取。

## 临时障碍物

```bash
./start_tracking.sh --map-json src/simulator/map/maze1.json \
  --temporary-obstacles --obstacle-seed 42 --direct-target
```

每 5 秒尝试新增一个高 4 m 的圆柱、箱体或窄墙；RViz 的 `Publish Point` 也可追加。只在机器人发生有效平移时生成，连续 3 秒不平移会暂停；障碍保留至本次进程退出，不修改源地图。候选位置避开静态障碍、边界及两个机器人。固定种子控制随机序列，ROS 异步时序仍可能改变实际放置结果。

## 主要话题

| 话题 | 内容 |
| --- | --- |
| `/map` | 全局占据图，用于显示及实验工具 |
| `/tracker/odom` | 跟踪机器人里程计 |
| `/tracker/scan` | 360°点云，供 tracker 构建滚动局部地图 |
| `/tracker/map3d` | 三维障碍点云 |
| `/tracker/fov` | 相机视锥 |
| `/target/detection` | 目标检测 |
| `/target/ground_truth` | 仿真目标真值 |
| `/goal_pose` | 目标导航点输入 |
| `/target/reference_pose` | 外部目标轨迹输入，需启用外部控制 |
