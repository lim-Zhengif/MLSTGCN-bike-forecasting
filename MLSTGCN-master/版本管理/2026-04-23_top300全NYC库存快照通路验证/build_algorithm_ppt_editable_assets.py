import json
import os
import zipfile
from html import escape


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(PROJECT_ROOT, "分析结果", "2026-04-24_阶段汇报PPT材料_算法代码版")

SLIDE_W = 12192000
SLIDE_H = 6858000


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def emu_x(value):
    return int(value * SLIDE_W)


def emu_y(value):
    return int(value * SLIDE_H)


def p_xml(text, size=1800, color="1F2937", bold=False):
    text = "" if text is None else str(text)
    lines = text.split("\n")
    parts = []
    for line in lines:
        safe = escape(line)
        b = ' b="1"' if bold else ""
        parts.append(
            f'<a:p><a:r><a:rPr lang="zh-CN" sz="{size}"{b}>'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/>'
            f'</a:rPr><a:t>{safe}</a:t></a:r><a:endParaRPr lang="zh-CN" sz="{size}"/></a:p>'
        )
    return "".join(parts)


def shape_xml(idx, x, y, w, h, text, fill="FFFFFF", line="CBD5E1", size=1700, color="1F2937", bold=False, radius=False):
    geom = "roundRect" if radius else "rect"
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{idx}" name="Shape {idx}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{emu_x(x)}" y="{emu_y(y)}"/><a:ext cx="{emu_x(w)}" cy="{emu_y(h)}"/></a:xfrm>
        <a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>
        <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>
        <a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>
      </p:spPr>
      <p:txBody><a:bodyPr wrap="square" lIns="91440" tIns="68580" rIns="91440" bIns="68580"/><a:lstStyle/>{p_xml(text, size, color, bold)}</p:txBody>
    </p:sp>
    """


def textbox_xml(idx, x, y, w, h, text, size=1800, color="1F2937", bold=False):
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{idx}" name="TextBox {idx}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{emu_x(x)}" y="{emu_y(y)}"/><a:ext cx="{emu_x(w)}" cy="{emu_y(h)}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/><a:ln><a:noFill/></a:ln>
      </p:spPr>
      <p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"/><a:lstStyle/>{p_xml(text, size, color, bold)}</p:txBody>
    </p:sp>
    """


def line_xml(idx, x1, y1, x2, y2, color="64748B"):
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return f"""
    <p:cxnSp>
      <p:nvCxnSpPr><p:cNvPr id="{idx}" name="Connector {idx}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{emu_x(x)}" y="{emu_y(y)}"/><a:ext cx="{emu_x(w)}" cy="{emu_y(h)}"/></a:xfrm>
        <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
        <a:ln w="19050"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:tailEnd type="triangle"/></a:ln>
      </p:spPr>
    </p:cxnSp>
    """


def table_like(slide_id, x, y, w, row_h, headers, rows):
    parts = []
    cols = len(headers)
    col_w = w / cols
    idx = slide_id
    for c, header in enumerate(headers):
        parts.append(shape_xml(idx, x + c * col_w, y, col_w, row_h, header, fill="DBEAFE", line="93C5FD", size=1400, color="1E3A8A", bold=True))
        idx += 1
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            parts.append(shape_xml(idx, x + c * col_w, y + (r + 1) * row_h, col_w, row_h, cell, fill="FFFFFF", line="E2E8F0", size=1280, color="334155"))
            idx += 1
    return "".join(parts), idx


SLIDES = [
    {
        "title": "算法与代码实现思路汇报版",
        "subtitle": "共享单车小时级骑入/骑出预测、库存风险识别与调度仿真",
        "boxes": [
            ("本版目的", "不只展示结果图，而是讲清楚“数据如何进来、模型如何预测、预测如何变成调度输入”。"),
            ("当前状态", "旧JC 105站已完成预测-库存-调度链路；新NYC Top300已完成数据准备，训练受显存限制，下一步转Top150。"),
        ],
    },
    {
        "title": "整体算法链路",
        "flow": [
            ("订单数据", "骑行起终点\n小时聚合"),
            ("库存快照", "站点容量\n可用车/空桩"),
            ("TopK筛站", "库存可匹配\n按全年流量排序"),
            ("多图构建", "距离/OD/相关性\n语义图"),
            ("预测模型", "历史168小时\n预测未来24小时"),
            ("调度输入", "库存轨迹\n风险标签\n建议调度量"),
        ],
        "note": "这页建议作为方法章节总图，所有框和文字都可以直接在PPT中修改。",
    },
    {
        "title": "数据处理一：站点匹配与TopK筛选",
        "left": "代码入口：build_topk_nyc_inventory_assets.py\n核心问题：订单station_id与GBFS库存station_id不是同一体系，不能直接按ID连接。\n解决方式：以站点名称为主匹配，必要时用经纬度和容量做校验。",
        "right": "筛选规则：\n1. 读取全年订单分片和库存快照\n2. 保留能匹配库存快照的站点\n3. 统计全年 total_start_count + total_end_count\n4. 按总流量降序选择 TopK\n5. 输出 GNN_Node_Mapping_topk.csv",
        "formula": "score_i = start_count_i + end_count_i",
    },
    {
        "title": "数据处理二：小时级样本构造",
        "left": "代码入口：prepare_topk_hourly_dataset.py + hourly_pipeline_utils.py\n输入：TopK站点映射、全年订单、天气、日历、站点容量。\n输出：train/val/test.npz。",
        "right": "样本定义：\n历史窗口：过去168小时\n预测窗口：未来24小时\n目标变量：每站每小时骑出量、骑入量\n已知未来特征：capacity、星期、是否周末、节假日、天气",
        "formula": "X_t = [t-167, ..., t]，Y_t = [t+1, ..., t+24]",
    },
    {
        "title": "多图构建思路",
        "cards": [
            ("空间距离图", "基于站点经纬度，距离越近权重越高。"),
            ("OD转移图", "统计站点间真实骑行流向，描述客流迁移。"),
            ("需求相关图", "用日净流量相关性刻画需求同步关系。"),
            ("语义/功能图", "用容量、经纬度等静态属性构造站点相似性。"),
        ],
        "formula": "A = {A_dist, A_od, A_corr, A_semantic, A_temporal}",
    },
    {
        "title": "exp06模型配置与迁移实验",
        "headers": ["配置项", "当前取值", "作用"],
        "rows": [
            ["历史窗口", "168小时", "捕捉一周周期"],
            ["预测窗口", "24小时", "支持全天与高峰调度"],
            ["损失函数", "Huber(delta=2)", "降低异常流量冲击"],
            ["学习率", "5e-5", "旧实验中较稳定"],
            ["图稀疏", "TopK=20", "保留重要邻接关系"],
            ["图注意力", "开启", "动态融合多张图"],
        ],
    },
    {
        "title": "多图注意力为什么会卡显存",
        "left": "当前OOM不是路径错误，而是Top300节点下图注意力张量过大。\nFusionGraph中注意力近似和节点数平方相关，因此单纯降低batch size只能缓解一部分。",
        "right": "规模估计：\nTop300作为1.00\nTop200约0.44\nTop150约0.25\n所以从数据处理层面先跑Top150，是更稳的下一步。",
        "formula": "memory ≈ O(N² × 图数量 × 注意力头数)",
    },
    {
        "title": "预测结果如何进入库存风险",
        "left": "模型输出不是直接调度，而是先转成库存轨迹。\n每个站点从当前库存出发，按未来每小时预测骑入/骑出滚动更新。",
        "right": "风险判断：\n低于S_min：缺车风险\n高于S_max：满桩风险\n记录风险开始时间、越界量、连续越界小时\n最终形成 dispatch_decision_input.csv",
        "formula": "S_{t+h} = S_t + Σ(pred_in - pred_out)",
    },
    {
        "title": "调度仿真接口",
        "left": "调度策略不直接看MAE，而是读取统一调度输入表。\n表中包含预测流量、预测库存、安全区间、风险标签、建议调入/调出量。",
        "right": "策略逻辑：\n调入站：未来低于S_min且持续短缺\n调出站：当前高于S_max且未来仍富余\n每小时可重算，但不代表每小时都必须调度",
        "formula": "dispatch_input = prediction + inventory + risk + suggested_action",
    },
    {
        "title": "下一步实验安排",
        "headers": ["阶段", "目标", "产出"],
        "rows": [
            ["Top150", "先跑通NYC高流量小规模模型", "训练结果和误差分布"],
            ["Top200", "验证规模扩大后的稳定性", "与Top150对比"],
            ["Top300轻量版", "保留更大站点覆盖", "关闭注意力或降低heads"],
            ["调度/RL", "从规则调度过渡到策略学习", "状态、动作、奖励设计"],
        ],
    },
]


def slide_xml(slide, slide_num):
    shape_id = 2
    bg = '<p:bg><p:bgPr><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>'
    parts = [textbox_xml(shape_id, 0.055, 0.045, 0.89, 0.07, slide["title"], size=2700, color="0F172A", bold=True)]
    shape_id += 1
    parts.append(shape_xml(shape_id, 0.055, 0.13, 0.89, 0.012, "", fill="2563EB", line="2563EB"))
    shape_id += 1

    if "subtitle" in slide:
        parts.append(textbox_xml(shape_id, 0.065, 0.19, 0.86, 0.09, slide["subtitle"], size=1900, color="475569"))
        shape_id += 1
        for i, (title, body) in enumerate(slide["boxes"]):
            parts.append(shape_xml(shape_id, 0.10 + i * 0.42, 0.36, 0.36, 0.28, f"{title}\n{body}", fill="FFFFFF", line="93C5FD", size=1550, color="1E293B", bold=False, radius=True))
            shape_id += 1

    if "flow" in slide:
        x0, w, gap = 0.055, 0.135, 0.024
        for i, (title, body) in enumerate(slide["flow"]):
            x = x0 + i * (w + gap)
            parts.append(shape_xml(shape_id, x, 0.31, w, 0.23, f"{title}\n{body}", fill="FFFFFF", line="60A5FA", size=1280, color="1E293B", bold=False, radius=True))
            shape_id += 1
            if i < len(slide["flow"]) - 1:
                parts.append(line_xml(shape_id, x + w, 0.425, x + w + gap * 0.75, 0.425))
                shape_id += 1
        parts.append(textbox_xml(shape_id, 0.08, 0.68, 0.84, 0.08, slide["note"], size=1550, color="475569"))
        shape_id += 1

    if "left" in slide and "right" in slide:
        parts.append(shape_xml(shape_id, 0.07, 0.21, 0.39, 0.40, slide["left"], fill="FFFFFF", line="CBD5E1", size=1450, color="1F2937", radius=True))
        shape_id += 1
        parts.append(shape_xml(shape_id, 0.54, 0.21, 0.39, 0.40, slide["right"], fill="FFFFFF", line="CBD5E1", size=1450, color="1F2937", radius=True))
        shape_id += 1
        parts.append(shape_xml(shape_id, 0.13, 0.68, 0.74, 0.12, slide["formula"], fill="EFF6FF", line="93C5FD", size=1800, color="1D4ED8", bold=True, radius=True))
        shape_id += 1

    if "cards" in slide:
        for i, (title, body) in enumerate(slide["cards"]):
            x = 0.08 + (i % 2) * 0.44
            y = 0.22 + (i // 2) * 0.24
            parts.append(shape_xml(shape_id, x, y, 0.38, 0.17, f"{title}\n{body}", fill="FFFFFF", line="93C5FD", size=1450, color="1E293B", radius=True))
            shape_id += 1
        parts.append(shape_xml(shape_id, 0.13, 0.72, 0.74, 0.11, slide["formula"], fill="F8FAFC", line="CBD5E1", size=1750, color="1D4ED8", bold=True, radius=True))
        shape_id += 1

    if "headers" in slide:
        table, next_id = table_like(shape_id, 0.075, 0.20, 0.85, 0.085, slide["headers"], slide["rows"])
        parts.append(table)
        shape_id = next_id

    parts.append(textbox_xml(shape_id, 0.86, 0.91, 0.08, 0.04, f"{slide_num}", size=1200, color="94A3B8"))
    sp_tree = "".join(parts)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>{bg}<p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {sp_tree}
  </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def write_pptx(path):
    slide_count = len(SLIDES)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        overrides = [
            '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
            '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
            '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
            '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        ]
        for i in range(1, slide_count + 1):
            overrides.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
        z.writestr("[Content_Types].xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {"".join(overrides)}
</Types>""")
        z.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>""")
        sld_ids = "".join([f'<p:sldId id="{255+i}" r:id="rId{i}"/>' for i in range(1, slide_count + 1)])
        z.writestr("ppt/presentation.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{slide_count + 1}"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>""")
        rels = []
        for i in range(1, slide_count + 1):
            rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>')
        rels.append(f'<Relationship Id="rId{slide_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>')
        z.writestr("ppt/_rels/presentation.xml.rels", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(rels)}</Relationships>""")
        for i, slide in enumerate(SLIDES, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(slide, i))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>""")
        z.writestr("ppt/slideMasters/slideMaster1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>""")
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>""")
        z.writestr("ppt/slideLayouts/slideLayout1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>""")
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>""")
        z.writestr("ppt/theme/theme1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="AlgorithmReport">
  <a:themeElements>
    <a:clrScheme name="Default"><a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="F97316"/></a:accent2><a:accent3><a:srgbClr val="22C55E"/></a:accent3><a:accent4><a:srgbClr val="64748B"/></a:accent4><a:accent5><a:srgbClr val="0EA5E9"/></a:accent5><a:accent6><a:srgbClr val="A855F7"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Default"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Default"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>""")


def write_markdown_and_json():
    md_lines = ["# 算法/代码思路版PPT讲稿", ""]
    for i, slide in enumerate(SLIDES, start=1):
        md_lines.append(f"## 第{i}页：{slide['title']}")
        if "subtitle" in slide:
            md_lines.append(slide["subtitle"])
        if "left" in slide:
            md_lines.append(slide["left"])
            md_lines.append(slide["right"])
            md_lines.append(f"公式/口径：`{slide['formula']}`")
        if "flow" in slide:
            md_lines.append("流程：" + " -> ".join([x[0] for x in slide["flow"]]))
        if "cards" in slide:
            for title, body in slide["cards"]:
                md_lines.append(f"- {title}：{body}")
        if "headers" in slide:
            md_lines.append("表格内容已放入PPT原生表格样式文本框，可直接修改。")
        md_lines.append("")

    mermaid = """# 可复制到支持Mermaid的工具中修改

```mermaid
flowchart LR
  A[全年订单数据] --> B[站点名称匹配库存快照]
  B --> C[按总骑入骑出筛选TopK]
  C --> D[构建距离图/OD图/需求相关图/语义图]
  D --> E[MLSTGCN exp06预测未来24小时骑入骑出]
  E --> F[预测流量累计为库存轨迹]
  F --> G[S_min/S_max风险标签]
  G --> H[dispatch_decision_input统一调度输入]
  H --> I[规则仿真或强化学习环境]
```

```mermaid
flowchart TB
  X[历史168小时 X] --> M[MSTGCN时空卷积]
  G1[空间距离图] --> F[多图融合/注意力]
  G2[OD转移图] --> F
  G3[需求相关图] --> F
  G4[语义图] --> F
  F --> M
  M --> Y[未来24小时骑出/骑入 Y]
```
"""
    with open(os.path.join(OUT_DIR, "算法代码版_逐页讲稿.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines))
    with open(os.path.join(OUT_DIR, "算法代码版_Mermaid流程图.md"), "w", encoding="utf-8") as handle:
        handle.write(mermaid)
    with open(os.path.join(OUT_DIR, "算法代码版_页面内容.json"), "w", encoding="utf-8") as handle:
        json.dump(SLIDES, handle, ensure_ascii=False, indent=2)


def main():
    ensure_dir(OUT_DIR)
    pptx_path = os.path.join(OUT_DIR, "算法代码版_可编辑PPT.pptx")
    write_pptx(pptx_path)
    write_markdown_and_json()
    print("Saved editable algorithm report assets to:", OUT_DIR)
    print("PPTX:", pptx_path)


if __name__ == "__main__":
    main()
