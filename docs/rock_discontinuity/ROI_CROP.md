# 河流上方裸岩 ROI 裁剪

原始 `cloud.las` 没有 RGB，不能根据水体颜色可靠地区分河流和裸岩。本项目使用 ENU
斜投影多边形限定分析范围：保留河流上方岩坡，排除河面、河床和道路。ROI 只是分析
边界，不是节理面人工标注。

## 1. 流式裁剪

```bash
cd "/home/huqilin/项目/节理面"

osgb_pipeline/.venv/bin/python main.py crop-roi \
  --input "/media/huqilin/新加卷/节理面数据/outputs/root_full_xyz_10/cloud.las" \
  --roi-config "rock_discontinuity/config/river_upper_cliff_roi.json" \
  --output "/media/huqilin/新加卷/节理面数据/outputs/river_upper_cliff/cloud.laz" \
  --report "/media/huqilin/新加卷/节理面数据/outputs/river_upper_cliff/crop_report.json" \
  --chunk-size 1000000
```

处理按块读取，不把 14.75 亿点加载到内存；输出保留源文件的点格式、比例、偏移、CRS
和所有原始维度。程序先写入 `.part` 临时文件，完成后才原子替换正式输出。

## 2. 只对岩坡识别节理面

裁剪完成后执行：

```bash
cd "/home/huqilin/项目/节理面"

osgb_pipeline/.venv/bin/python main.py whole-cloud \
  --input "/media/huqilin/新加卷/节理面数据/outputs/river_upper_cliff/cloud.laz" \
  --output "/media/huqilin/新加卷/节理面数据/outputs/river_upper_cliff_joint_planes" \
  --config "rock_discontinuity/config/default.yaml" \
  --tile-size 25 \
  --overlap 1 \
  --workers 2
```

## 3. 输出检查

裁剪报告需要满足：

- `status=complete`
- `output.point_count > 0`
- `0 < output.selected_fraction < 1`
- `discarded_nonfinite_points=0`
- `preservation.all_source_dimensions_preserved=true`

若要调整岩坡边界，只修改 `river_upper_cliff_roi.json` 中的 `polygon_uv`，不要修改原始
LAS。正式运行前应核对同一 ROI 配置生成的蓝色边界预览。
