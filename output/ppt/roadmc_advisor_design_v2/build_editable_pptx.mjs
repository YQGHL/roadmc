import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const pptxgen = require("C:/Users/SEELE/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = "C:/Users/SEELE/PycharmProjects/PythonProject";
const outPath = path.join(root, "output/ppt/roadmc_advisor_for_teacher_editable_v1.pptx");

const pptx = new pptxgen();
pptx.author = "Codex";
pptx.company = "RoadMC";
pptx.subject = "RoadMC 导师汇报可编辑版";
pptx.title = "RoadMC 导师汇报可编辑版";
pptx.lang = "zh-CN";
pptx.defineLayout({ name: "ROADMC_WIDE", width: 13.333333, height: 7.5 });
pptx.layout = "ROADMC_WIDE";
pptx.theme = {
  headFontFace: "Noto Serif SC",
  bodyFontFace: "Noto Sans SC",
  lang: "zh-CN",
};

const W = 13.333333;
const H = 7.5;
const C = {
  paper: "F5EADC",
  paperLight: "FBF6EE",
  ink: "211915",
  text: "3B2C24",
  muted: "78675B",
  line: "C8AD96",
  lineSoft: "E3D2C1",
  accent: "A9472C",
  accentSoft: "DC8B67",
  green: "526D61",
  blue: "465F76",
  grid: "EADFD2",
  white: "FFFFFF",
};
const F = {
  serif: "Noto Serif SC",
  sans: "Noto Sans SC",
  display: "Styrene A Web",
  tiempos: "Tiempos Text",
};

const notes = [
  "开场先讲定位：RoadMC 不是一个单独分割模型，而是围绕路面病害点云任务建立“生成、训练、评估”的闭环。",
  "这一页只做问题框定。重点是说明真实点云监督成本、病害稀疏和长尾类别会直接决定路线设计。",
  "这页把生成模块提升为方法本体。强调生成不是临时造数据，而是控制几何、标签和监督上限。",
  "讲清楚点云如何生成：路面形貌、病害原语、传感器观测，最后输出可训练的 npz 样本。",
  "解释为什么先做二分类。它不是放弃 38 类，而是先稳定前景边界，再迁移到更细粒度类别。",
  "主动讲证据边界。当前结果能证明路线可行，但仍基于合成小型测试集，不能包装成最终泛化结论。",
  "GAN 和 end2end 要定位为未来真实域适配接口，不要把它们说成当前主结果。",
  "这页主动暴露风险，便于导师判断下一步验证优先级。",
  "收尾用两个路线选择引导讨论：论文路线与展示路线。请导师判断优先推进哪一种。",
];

function addBase(slide, n) {
  slide.background = { color: C.paper };
  for (let x = 0; x <= W; x += 0.35) {
    slide.addShape(pptx.ShapeType.line, {
      x, y: 0, w: 0, h: H,
      line: { color: C.grid, width: 0.35, transparency: 36 },
    });
  }
  for (let y = 0; y <= H; y += 0.35) {
    slide.addShape(pptx.ShapeType.line, {
      x: 0, y, w: W, h: 0,
      line: { color: C.grid, width: 0.35, transparency: 36 },
    });
  }
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.28, y: 0.25, w: 12.78, h: 7.0,
    fill: { color: C.paper, transparency: 100 },
    line: { color: C.lineSoft, width: 0.6, transparency: 10 },
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 11.78, y: 7.0, w: 0.6, h: 0,
    line: { color: C.line, width: 0.8, transparency: 6 },
  });
  addText(slide, n, 12.55, 6.91, 0.25, 0.18, {
    fontFace: F.display, fontSize: 7.5, color: C.muted, align: "right",
  });
}

function addText(slide, text, x, y, w, h, opt = {}) {
  slide.addText(text, {
    x, y, w, h,
    margin: opt.margin ?? 0,
    fontFace: opt.fontFace ?? F.sans,
    fontSize: opt.fontSize ?? 14,
    color: opt.color ?? C.ink,
    bold: opt.bold ?? false,
    italic: opt.italic ?? false,
    breakLine: false,
    fit: "shrink",
    valign: opt.valign ?? "top",
    align: opt.align ?? "left",
    paraSpaceAfterPt: opt.paraSpaceAfterPt ?? 0,
    paraSpaceBeforePt: 0,
    breakLine: false,
    isTextBox: true,
  });
}

function addSection(slide, n, label) {
  addText(slide, n, 0.78, 0.55, 0.32, 0.2, {
    fontFace: F.display, fontSize: 11, color: C.accent,
  });
  addText(slide, label, 1.35, 0.56, 2.8, 0.18, {
    fontFace: F.display, fontSize: 9.6, color: C.accent, bold: true,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 1.35, y: 0.71, w: 3.9, h: 0,
    line: { color: C.line, width: 0.55, transparency: 10 },
  });
}

function addTitle(slide, title, y = 1.02, w = 8.8, size = 34) {
  addText(slide, title, 0.78, y, w, 0.92, {
    fontFace: F.serif, fontSize: size, color: C.ink, bold: true,
  });
}

function addLead(slide, text, y = 2.02, w = 8.1) {
  addText(slide, text, 0.78, y, w, 0.62, {
    fontFace: F.sans, fontSize: 14.5, color: C.muted,
  });
}

function addRule(slide, x, y, w, width = 0.65, color = C.line) {
  slide.addShape(pptx.ShapeType.line, {
    x, y, w, h: 0,
    line: { color, width, transparency: 4 },
  });
}

function addRect(slide, x, y, w, h, opt = {}) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h,
    fill: { color: opt.fill ?? C.paperLight, transparency: opt.fillTrans ?? 18 },
    line: { color: opt.line ?? C.line, width: opt.lineWidth ?? 0.65, transparency: opt.lineTrans ?? 6 },
    radius: opt.radius ?? 0,
  });
}

function addArrow(slide, x, y, w, h = 0) {
  slide.addShape(pptx.ShapeType.line, {
    x, y, w, h,
    line: { color: C.accentSoft, width: 1.2, endArrowType: "triangle" },
  });
}

function addCircle(slide, x, y, r, fill = C.blue, trans = 25) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x: x - r, y: y - r, w: r * 2, h: r * 2,
    fill: { color: fill, transparency: trans },
    line: { color: fill, transparency: 100 },
  });
}

function addPolyline(slide, pts, color = C.ink, width = 1.25) {
  for (let i = 1; i < pts.length; i += 1) {
    const a = pts[i - 1];
    const b = pts[i];
    slide.addShape(pptx.ShapeType.line, {
      x: a[0], y: a[1], w: b[0] - a[0], h: b[1] - a[1],
      line: { color, width },
    });
  }
}

function note(slide, i) {
  if (typeof slide.addNotes === "function") slide.addNotes(notes[i]);
}

function slide1() {
  const s = pptx.addSlide();
  addBase(s, "01");
  addText(s, "MENTOR RESEARCH BRIEFING / 2026", 0.78, 0.63, 3.2, 0.22, {
    fontFace: F.display, fontSize: 10, color: C.accent,
  });
  addText(s, "RoadMC", 0.78, 1.72, 4.9, 0.82, {
    fontFace: F.display, fontSize: 78, color: C.ink, bold: true,
  });
  addText(s, "物理仿真点云生成与路面病害分割训练闭环", 0.78, 2.72, 5.9, 0.48, {
    fontFace: F.serif, fontSize: 21, color: C.text,
  });
  addText(s, "核心不是单个网络，而是一套从可控数据生成到病害分割训练、\n再到评估校准的研究闭环。", 0.78, 5.92, 6.2, 0.66, {
    fontFace: F.sans, fontSize: 17, color: C.ink,
  });
  addRule(s, 8.6, 1.07, 3.95, 1.3, C.ink);
  addText(s, "这次汇报要回答的问题", 8.6, 1.28, 3.9, 0.36, {
    fontFace: F.serif, fontSize: 16.5, color: C.ink,
  });
  const rows = ["监督数据从哪里来", "模型路线怎样稳定", "结果边界如何解释", "下一步验证怎么做"];
  rows.forEach((r, i) => {
    const y = 1.92 + i * 0.62;
    addRule(s, 8.6, y - 0.1, 3.95, 0.55, C.line);
    addText(s, String(i + 1).padStart(2, "0"), 8.6, y, 0.34, 0.16, {
      fontFace: F.display, fontSize: 9, color: C.accent,
    });
    addText(s, r, 9.12, y - 0.03, 2.85, 0.23, {
      fontFace: F.sans, fontSize: 15, color: C.ink, bold: true,
    });
  });
  addRule(s, 9.05, 5.72, 3.4, 0.55, C.line);
  addText(s, "Synthesis is treated as a first-class method\ncomponent, not a disposable preprocessing\nstep.", 9.05, 5.88, 3.35, 0.62, {
    fontFace: F.tiempos, fontSize: 14.5, color: C.muted,
  });
  note(s, 0);
}

function slide2() {
  const s = pptx.addSlide();
  addBase(s, "02");
  addSection(s, "02", "PROBLEM FRAMING");
  addTitle(s, "直接训练路面病害点云模型，很容易卡在数据和长尾上", 0.96, 8.9, 26);
  addLead(s, "如果直接从真实点云训练 38 类病害分割，最早遇到的通常不是网络结构上限，而是监督数据、标签成本和类别分布共同造成的不稳定。", 1.82, 7.4);
  const nodes = [
    ["A", "真实采集贵", "点云采集、清洗和逐点标注成本高，早期很难快速形成可控监督。"],
    ["B", "病害目标稀疏", "裂缝、剥落、坑槽等点占比低，模型容易偏向预测背景。"],
    ["C", "38 类长尾明显", "少数类别样本不足时，冷启动多分类训练容易塌缩。"],
  ];
  nodes.forEach((n, i) => {
    const x = 0.78 + i * 4.0;
    addRule(s, x, 3.05, 2.35, 1.05, i === 0 ? C.ink : C.line);
    addText(s, n[0], x, 3.25, 0.2, 0.18, { fontFace: F.display, fontSize: 10, color: C.accent, bold: true });
    addText(s, n[1], x, 3.78, 1.95, 0.32, { fontFace: F.serif, fontSize: 19, color: C.ink, bold: true });
    addText(s, n[2], x, 4.38, 2.25, 0.72, { fontFace: F.sans, fontSize: 12.5, color: C.muted });
    if (i < 2) addArrow(s, x + 2.55, 3.5, 1.05, 0);
  });
  addRule(s, 0.78, 6.22, 10.1, 0.7, C.line);
  addText(s, "讲法", 0.78, 6.42, 0.55, 0.2, { fontFace: F.sans, fontSize: 11, color: C.accent, bold: true });
  addText(s, "RoadMC 的价值不只是换网络，而是先把“可控监督”和“稳定训练路线”建立起来。", 1.72, 6.39, 7.6, 0.3, {
    fontFace: F.sans, fontSize: 14.5, color: C.ink,
  });
  note(s, 1);
}

function slide3() {
  const s = pptx.addSlide();
  addBase(s, "03");
  addSection(s, "03", "METHOD POSITIONING");
  addTitle(s, "把点云生成提升为方法的一半，再用训练结果反向校准生成", 0.96, 8.9, 28);
  addRect(s, 0.88, 2.52, 3.65, 2.55, { fill: C.paperLight, fillTrans: 14, line: C.lineSoft });
  s.addShape(pptx.ShapeType.line, { x: 0.88, y: 2.52, w: 0, h: 2.55, line: { color: C.accent, width: 3 } });
  addText(s, "生成不是临时造数据，\n而是在定义监督空间。", 1.08, 2.76, 3.05, 0.8, {
    fontFace: F.serif, fontSize: 20.5, color: C.ink, bold: true,
  });
  addText(s, "生成模块决定几何形态、标签分布、噪声结构和训练可见的病害边界；训练模块验证这些监督是否真的能被模型学到。", 1.08, 3.78, 3.05, 0.82, {
    fontFace: F.sans, fontSize: 13.5, color: C.text,
  });
  addRect(s, 6.18, 2.05, 3.18, 3.18, { fill: C.paper, fillTrans: 100, line: C.line });
  addRect(s, 6.55, 2.42, 2.44, 2.44, { fill: C.paper, fillTrans: 100, line: C.accent, lineTrans: 55 });
  addRect(s, 7.08, 2.95, 1.38, 1.38, { fill: C.paperLight, fillTrans: 6, line: C.lineSoft });
  addText(s, "生成 / 训练 /\n评估闭环", 7.28, 3.28, 0.95, 0.42, {
    fontFace: F.serif, fontSize: 13.5, color: C.ink, align: "center",
  });
  const items = [
    [4.95, 1.58, "01", "物理先验", "粗糙度、微纹理、路面形貌"],
    [9.55, 1.58, "02", "病害几何", "裂缝、坑槽、车辙、剥落"],
    [9.55, 5.25, "03", "分割训练", "Swin3D / mHC / binary first"],
    [4.95, 5.25, "04", "评估校准", "mIoU、阈值、误差回看"],
  ];
  items.forEach(([x, y, num, h, b]) => {
    addRule(s, x, y, 1.88, 1.05, C.ink);
    addText(s, num, x, y + 0.16, 0.22, 0.12, { fontFace: F.display, fontSize: 8, color: C.accent, bold: true });
    addText(s, h, x, y + 0.45, 1.5, 0.22, { fontFace: F.serif, fontSize: 14, color: C.ink, bold: true });
    addText(s, b, x, y + 0.78, 1.65, 0.34, { fontFace: F.sans, fontSize: 9.5, color: C.muted });
  });
  note(s, 2);
}

function slide4() {
  const s = pptx.addSlide();
  addBase(s, "04");
  addSection(s, "04", "POINT CLOUD SYNTHESIS");
  addTitle(s, "合成点云来自一条物理和几何流水线，不是随机造点", 0.96, 9.1, 34);
  addLead(s, "生成流程按“路面形貌、病害形变、传感器观测”组织，最后输出带逐点标签的训练样本。", 2.28, 7.4);
  addRule(s, 0.78, 3.08, 7.6, 0.65, C.line);
  const rows = [
    ["01", "路面形貌", "用 ISO 8608 粗糙度谱生成宏观起伏，再叠加 fBm 微纹理和局部法向变化。"],
    ["02", "病害形变", "通过裂缝、坑槽、车辙、剥落等病害原语写入几何结构，标签从几何区域映射得到。"],
    ["03", "LiDAR 观测", "用扫描线重采样、距离噪声和点密度控制，让输入更接近真实扫描点云。"],
  ];
  rows.forEach((r, i) => {
    const y = 3.58 + i * 1.38;
    addRule(s, 0.78, y + 0.86, 7.6, 0.55, C.line);
    addText(s, r[0], 0.78, y, 0.55, 0.38, { fontFace: F.display, fontSize: 21, color: C.accent, bold: true });
    addText(s, r[1], 1.75, y, 2.1, 0.32, { fontFace: F.serif, fontSize: 20, color: C.ink, bold: true });
    addText(s, r[2], 4.48, y - 0.03, 3.1, 0.5, { fontFace: F.sans, fontSize: 12.8, color: C.muted });
  });
  addRect(s, 8.92, 3.12, 3.75, 2.42, { fill: C.paperLight, fillTrans: 40, line: C.lineSoft });
  addText(s, "GEOMETRY + SENSOR OBSERVATION", 9.36, 3.58, 2.9, 0.18, {
    fontFace: F.display, fontSize: 8.5, color: C.accent,
  });
  addRule(s, 9.36, 3.85, 2.75, 0.45, C.line);
  addPolyline(s, [[9.45, 4.55], [9.98, 4.58], [10.45, 4.48], [10.95, 4.43], [11.45, 4.53], [12.12, 4.45]], C.ink, 1.45);
  addPolyline(s, [[9.92, 4.58], [10.02, 4.24], [10.22, 4.02], [10.42, 4.28], [10.52, 4.58]], C.accent, 2);
  addPolyline(s, [[11.05, 4.52], [11.26, 4.35], [11.58, 4.34], [11.82, 4.49]], C.accent, 2);
  [[9.62,4.72],[9.9,4.69],[10.16,4.76],[10.42,4.68],[10.72,4.63],[11.02,4.67],[11.36,4.78],[11.62,4.85],[11.88,4.73]].forEach(([x,y]) => addCircle(s, x, y, 0.027, C.blue, 25));
  addRule(s, 8.92, 5.78, 3.75, 1.0, C.ink);
  addText(s, ".npz output", 8.92, 5.98, 1.2, 0.2, { fontFace: F.display, fontSize: 12, color: C.accent, bold: true });
  ["points", "labels", "feats", "normals", "pavement_type"].forEach((t, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 8.92 + col * 1.95;
    const y = 6.34 + row * 0.36;
    addText(s, t, x, y, 1.55, 0.18, { fontFace: F.tiempos, fontSize: 12.5, color: C.ink });
    addRule(s, x, y + 0.26, 1.75, 0.45, C.lineSoft);
  });
  note(s, 3);
}

function slide5() {
  const s = pptx.addSlide();
  addBase(s, "05");
  addSection(s, "05", "TRAINING ROUTE");
  addTitle(s, "训练先稳定“病害前景”，再迁移到 38 类细分", 0.96, 9.35, 33);
  addLead(s, "当前更稳的路线是先做二分类，校准前景阈值和边界，再把 backbone 迁移到细粒度类别体系。", 2.03, 8.1);
  addRule(s, 0.78, 2.96, 2.7, 1.0, C.ink);
  const stages = [
    ["01", "二分类", "背景 / 病害，先解决能否识别前景。"],
    ["02", "阈值校准", "当前经验阈值约在 0.44 附近更优。"],
    ["03", "迁移骨干", "复用前景边界和几何表征。"],
    ["04", "38 类细分", "分阶段扩展，避免冷启动塌缩。"],
  ];
  stages.forEach((st, i) => {
    const y = 3.25 + i * 1.08;
    addText(s, st[0], 0.78, y, 0.3, 0.16, { fontFace: F.display, fontSize: 10, color: C.accent, bold: true });
    addText(s, st[1], 1.42, y - 0.02, 1.65, 0.28, { fontFace: F.sans, fontSize: 17, color: C.ink, bold: true });
    addText(s, st[2], 1.42, y + 0.36, 1.85, 0.42, { fontFace: F.sans, fontSize: 11.5, color: C.muted });
    addRule(s, 0.78, y + 0.84, 2.7, 0.45, C.lineSoft);
  });
  addText(s, "MODEL PIPELINE", 3.98, 2.98, 1.24, 0.2, { fontFace: F.display, fontSize: 9, color: C.accent, bold: true });
  addRect(s, 3.98, 3.42, 1.38, 0.92, { fill: C.paperLight, fillTrans: 18, line: C.line });
  addRect(s, 5.82, 3.42, 1.6, 0.92, { fill: C.paperLight, fillTrans: 18, line: C.line });
  addRect(s, 7.88, 3.42, 1.32, 0.92, { fill: C.paperLight, fillTrans: 18, line: C.line });
  addRect(s, 9.65, 3.42, 1.6, 0.92, { fill: C.paperLight, fillTrans: 18, line: C.line });
  [["输入点云","points / feats /\nnormals",3.98],["Swin3D","当前稳定精度基线",5.82],["mHC","通道混合增强",7.88],["Seg Head","二分类 / 多分类输出",9.65]].forEach(([h,b,x]) => {
    addText(s, h, x + 0.18, 3.62, 1.1, 0.22, { fontFace: F.serif, fontSize: 15.5, color: C.ink, bold: true });
    addText(s, b, x + 0.18, 3.96, 1.08, 0.26, { fontFace: F.sans, fontSize: 9.5, color: C.muted });
  });
  addArrow(s, 5.48, 3.88, 0.3);
  addArrow(s, 7.54, 3.88, 0.3);
  addArrow(s, 9.32, 3.88, 0.3);
  [["Focal","抑制背景主导"],["Dice","稳定小目标"],["Edge","强化裂缝边界"],["Muon + AdamW","混合优化，可回退"]].forEach((l, i) => {
    const x = 3.98 + i * 2.2;
    addRect(s, x, 4.72, 2.08, 0.78, { fill: C.paperLight, fillTrans: 14, line: C.accent, lineWidth: 0, lineTrans: 100 });
    addRule(s, x, 4.72, 2.08, 1.0, C.accent);
    addText(s, l[0], x + 0.16, 4.92, 1.4, 0.18, { fontFace: F.sans, fontSize: 10.5, color: C.accent, bold: true });
    addText(s, l[1], x + 0.16, 5.18, 1.4, 0.18, { fontFace: F.sans, fontSize: 9.5, color: C.muted });
  });
  note(s, 4);
}

function slide6() {
  const s = pptx.addSlide();
  addBase(s, "06");
  addSection(s, "06", "CURRENT EVIDENCE");
  addTitle(s, "当前结果说明路线可行，但不能被说成最终泛化结论", 0.96, 8.9, 27);
  const metrics = [
    [0.78, 2.42, 2.1, "0.7319", "Swin3D + mHC, 校准后 global mIoU", true],
    [3.08, 2.42, 2.1, "0.7052", "disease IoU", true],
    [0.78, 4.35, 1.75, "0.9039", "precision"],
    [2.62, 4.35, 1.75, "0.7624", "recall"],
    [4.46, 4.35, 1.75, "0.5671", "PointMamba + mHC 短跑 mIoU"],
    [6.3, 4.35, 1.75, "5000", "下一步目标合成场景规模"],
  ];
  metrics.forEach(([x,y,w,v,l,big]) => {
    addRule(s, x, y, w, big ? 1.25 : 0.65, big ? C.accent : C.line);
    addText(s, v, x, y + 0.18, w, 0.46, { fontFace: F.display, fontSize: big ? 31 : 22, color: big ? C.accent : C.ink, bold: true });
    addText(s, l, x, y + 0.82, w, 0.36, { fontFace: F.sans, fontSize: 9.5, color: C.muted });
    addRule(s, x, y + 1.32, w, 0.45, C.lineSoft);
  });
  addRule(s, 9.0, 2.35, 0, 0.65, C.line);
  s.addShape(pptx.ShapeType.line, { x: 8.9, y: 2.25, w: 0, h: 3.9, line: { color: C.line, width: 0.65 } });
  addText(s, "可以说：二分类路线已经\n从“能不能学”进入\n“如何提稳”。", 9.16, 2.35, 2.9, 0.92, {
    fontFace: F.serif, fontSize: 19, color: C.ink, bold: true,
  });
  addText(s, "Swin3D 当前更稳；PointMamba 是轻量候选；mHC 在当前主线中保留为通道混合模块。", 9.16, 3.58, 2.8, 0.65, {
    fontFace: F.sans, fontSize: 12, color: C.text,
  });
  addText(s, "边界：这些结果来自生成的小型验证 / 测试集，部分快速诊断 batch 很少，存在过拟合可能。", 9.16, 4.92, 2.8, 0.62, {
    fontFace: F.sans, fontSize: 11.8, color: C.accent,
  });
  note(s, 5);
}

function slide7() {
  const s = pptx.addSlide();
  addBase(s, "07");
  addSection(s, "07", "DOMAIN ADAPTATION BRANCH");
  addTitle(s, "GAN 和 end2end 是走向真实域的接口，不是当前主结果", 0.96, 8.7, 26);
  addLead(s, "代码中的 gan_enhanced / end2end 需要讲，但定位要准确：它们服务于合成到真实风格的域适配，主结果仍应先由稳定基线支撑。", 1.78, 8.1);
  addRule(s, 0.78, 2.85, 5.8, 1.0, C.ink);
  addText(s, "MAINLINE FIRST", 0.78, 3.04, 1.25, 0.16, { fontFace: F.display, fontSize: 8.5, color: C.accent, bold: true });
  const boxes = [
    [0.78, "合成点云", "可控几何和标签"],
    [2.65, "稳定训练", "Swin3D + mHC 二分类"],
    [4.8, "正式验证", "统一 global mIoU 口径"],
  ];
  boxes.forEach(([x,h,b], i) => {
    addRect(s, x, 3.65, 1.35, 0.82, { fill: C.paperLight, fillTrans: 20, line: C.line });
    addText(s, h, x + 0.15, 3.86, 1.05, 0.16, { fontFace: F.sans, fontSize: 10.5, color: C.ink, bold: true });
    addText(s, b, x + 0.15, 4.12, 1.0, 0.16, { fontFace: F.sans, fontSize: 8.2, color: C.muted });
    if (i < 2) addArrow(s, x + 1.48, 4.06, 0.34);
  });
  addRule(s, 7.28, 2.85, 3.6, 1.0, C.ink);
  addText(s, "FUTURE INTERFACE", 7.28, 3.04, 1.45, 0.16, { fontFace: F.display, fontSize: 8.5, color: C.accent, bold: true });
  const bullets = [
    "StyleTransferGen 对坐标和法向量做风格残差调整。",
    "WGAN 判别器判断点云风格是否更接近目标域。",
    "end2end 把风格调整与分割训练放入同一循环。",
    "主线稳定后，再用它验证合成到真实的域适配能力。",
  ];
  bullets.forEach((b, i) => {
    const y = 3.62 + i * 0.55;
    addText(s, b, 7.28, y, 3.2, 0.23, { fontFace: F.sans, fontSize: 11.5, color: C.text });
    addRule(s, 7.28, y + 0.34, 3.2, 0.42, C.lineSoft);
  });
  note(s, 6);
}

function slide8() {
  const s = pptx.addSlide();
  addBase(s, "08");
  addSection(s, "08", "RESEARCH RISKS");
  addTitle(s, "现在最该补的不是继续堆模块，而是扩大验证和统一口径", 0.96, 9.2, 28);
  addLead(s, "项目已经跑通主线，但要往论文或比赛成果推进，必须先解决可信度问题。", 1.86, 7.2);
  const risks = [
    ["01", "数据规模", "小型合成验证集容易让模型在少数 batch 上过拟合。", "扩大到约 5000 个合成场景，固定生成元数据。"],
    ["02", "评估口径", "batch mean mIoU 与全验证集 global mIoU 需要统一记录。", "保存 global mIoU、disease IoU、precision、recall。"],
    ["03", "38 类迁移", "从二分类直接跳到 38 类过陡，类别长尾会放大不稳定。", "二分类 → 高频子类 → 38 类，分阶段迁移。"],
    ["04", "真实域差异", "真实点云噪声、密度和采集角度仍未形成正式闭环。", "主线稳定后，再接 GAN / end2end 和真实点云验证。"],
  ];
  risks.forEach((r, i) => {
    const x = 0.78 + i * 3.0;
    addRule(s, x, 2.95, 2.42, 1.0, C.ink);
    addText(s, r[0], x, 3.22, 0.3, 0.16, { fontFace: F.display, fontSize: 9.5, color: C.accent, bold: true });
    addText(s, r[1], x, 3.86, 1.6, 0.28, { fontFace: F.serif, fontSize: 17.5, color: C.ink, bold: true });
    addText(s, r[2], x, 4.48, 2.05, 0.52, { fontFace: F.sans, fontSize: 11.2, color: C.muted });
    addRule(s, x, 5.42, 2.1, 0.45, C.line);
    addText(s, r[3], x, 5.62, 2.1, 0.52, { fontFace: F.sans, fontSize: 10.3, color: C.accent });
  });
  note(s, 7);
}

function slide9() {
  const s = pptx.addSlide();
  addBase(s, "09");
  addSection(s, "09", "ADVISOR DISCUSSION");
  addTitle(s, "下一阶段围绕“扩数据、稳基线、分阶段多类”推进", 0.96, 8.7, 28);
  const steps = [
    ["01", "扩大合成数据", "目标约 5000 个场景，优先保证可恢复生成、分批执行和统一元数据。"],
    ["02", "重做正式验证", "固定 Swin3D + mHC 二分类基线，统一 global mIoU、disease IoU、precision、recall。"],
    ["03", "分阶段回到 38 类", "二分类 → 高频子类 / 4-8 类 → 38 类，避免冷启动直接塌缩。"],
    ["04", "再接真实域", "主线稳定后，再接 GAN / end2end 和真实点云验证。"],
  ];
  addRule(s, 0.78, 2.35, 7.25, 1.0, C.ink);
  steps.forEach((st, i) => {
    const y = 2.66 + i * 0.77;
    addText(s, st[0], 0.78, y, 0.3, 0.16, { fontFace: F.display, fontSize: 9, color: C.accent, bold: true });
    addText(s, st[1], 1.45, y - 0.02, 1.85, 0.22, { fontFace: F.sans, fontSize: 12.2, color: C.ink, bold: true });
    addText(s, st[2], 4.15, y - 0.02, 3.5, 0.3, { fontFace: F.sans, fontSize: 9.6, color: C.muted });
    addRule(s, 0.78, y + 0.43, 7.25, 0.42, C.lineSoft);
  });
  addRect(s, 8.7, 2.28, 3.45, 3.55, { fill: C.paperLight, fillTrans: 16, line: C.line });
  addText(s, "希望导师判断的核心问题", 8.98, 2.58, 2.65, 0.4, {
    fontFace: F.serif, fontSize: 17.5, color: C.ink, bold: true,
  });
  addText(s, "这个项目下一步优先按论文路线推进，还是先按比赛 / 演示路线包装？两条路线需要不同的验证密度和交付重点。", 8.98, 3.18, 2.75, 0.72, {
    fontFace: F.sans, fontSize: 11.5, color: C.text,
  });
  addRule(s, 8.98, 4.3, 2.7, 0.55, C.line);
  addText(s, "A 论文路线：强调方法闭环、消融、数据规模和泛化验证。", 8.98, 4.52, 2.78, 0.28, {
    fontFace: F.sans, fontSize: 11, color: C.text,
  });
  addRule(s, 8.98, 4.9, 2.7, 0.55, C.line);
  addText(s, "B 展示路线：强调可视化、稳定 demo、生成样例和端到端流程。", 8.98, 5.12, 2.78, 0.28, {
    fontFace: F.sans, fontSize: 11, color: C.text,
  });
  note(s, 8);
}

[
  slide1,
  slide2,
  slide3,
  slide4,
  slide5,
  slide6,
  slide7,
  slide8,
  slide9,
].forEach((fn) => fn());

await pptx.writeFile({ fileName: outPath });
console.log(JSON.stringify({ outPath, slides: 9 }, null, 2));
