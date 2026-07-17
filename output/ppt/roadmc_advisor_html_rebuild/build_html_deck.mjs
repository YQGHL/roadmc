import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
const require = createRequire(import.meta.url);
const pptxgen = require("C:/Users/SEELE/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const outDir = "C:/Users/SEELE/PycharmProjects/PythonProject/output/ppt/roadmc_advisor_html_rebuild";
const htmlPath = path.join(outDir, "roadmc_advisor_html_deck.html");
const pngDir = path.join(outDir, "slides_png");
const pptxPath = "C:/Users/SEELE/PycharmProjects/PythonProject/output/ppt/roadmc_advisor_for_teacher_html_style.pptx";
const montagePath = path.join(outDir, "roadmc_advisor_for_teacher_montage.png");
const chromePath = "C:/Program Files/Google/Chrome/Application/chrome.exe";

const slides = [
  {
    label: "01",
    section: "项目定位",
    type: "cover",
    title: "RoadMC",
    subtitle: "物理仿真点云生成与路面病害分割训练闭环",
    thesis: "这次汇报不强调“我堆了多少模块”，而是回答：为什么要生成点云、模型现在证明了什么、下一步怎样把它推进成可靠成果。",
    points: ["数据从哪里来", "模型怎么学稳", "结果怎样解释", "下一步怎样验证"],
    notes: "开场先把定位讲清楚：RoadMC 不是单一分割网络，而是围绕路面病害点云任务建立生成、训练、评估的闭环。"
  },
  {
    label: "02",
    section: "为什么值得做",
    type: "problem",
    title: "难点不只是模型不够强，而是监督数据本身很难获得",
    lead: "如果直接从真实点云训练 38 类病害分割，最早遇到的通常不是网络结构上限，而是数据、标签和类别分布共同造成的不稳定。",
    blocks: [
      { k: "真实采集贵", t: "点云采集、清洗和点级标注成本高，早期很难快速形成大规模监督。" },
      { k: "病害很稀疏", t: "裂缝、剥落、坑槽等目标占比低，模型容易把多数点都预测成背景。" },
      { k: "类别长尾明显", t: "38 类细分任务冷启动过硬，少数类别缺样本会直接影响训练稳定性。" },
      { k: "采集域会变化", t: "扫描角度、点密度和噪声会改变输入分布，影响从合成到真实的迁移。" }
    ],
    bottom: "所以这个项目的主线不是“换一个更大的网络”，而是先把可控监督和训练路线建立起来。",
    notes: "这一页的目标是让导师先接受问题定义：RoadMC 的价值来自数据生成和训练路线，不只是模型结构。"
  },
  {
    label: "03",
    section: "总体思路",
    type: "loop",
    title: "把点云生成提升为方法的一半，再用训练结果反过来校准生成",
    lead: "RoadMC 的核心是闭环：生成模块提供可控监督，训练模块验证这些监督能否让模型学到病害边界。",
    nodes: [
      { h: "物理先验", b: "路面粗糙度与微纹理" },
      { h: "病害几何", b: "裂缝、坑槽、车辙、剥落" },
      { h: "LiDAR 观测", b: "扫描线、噪声、非均匀采样" },
      { h: "分割训练", b: "二分类稳定后再迁移多类别" },
      { h: "评估校准", b: "mIoU、阈值、误差回看" }
    ],
    callout: "要给导师强调：点云生成不是临时造数据，而是决定几何形态、标签分布和监督上限的方法模块。",
    notes: "讲这页时不要急着讲公式。重点是解释生成和训练为什么要放到一个系统里看。"
  },
  {
    label: "04",
    section: "点云生成模块",
    type: "threeLayer",
    title: "合成点云不是随机造点，而是三层结构叠加出来的训练样本",
    lead: "生成流程按“路面形貌 -> 病害形变 -> 传感器观测”组织，最后输出带逐点标签的 .npz 样本。",
    layers: [
      { h: "路面形貌", b: "用 ISO 8608 粗糙度谱生成宏观起伏，再叠加 fBm 微纹理和局部法向变化。" },
      { h: "病害形变", b: "在几何层面加入病害原语，标签从几何区域映射而来，不是后期随意贴标签。" },
      { h: "LiDAR 观测", b: "通过扫描线重采样、距离噪声和点密度控制，让数据更接近真实扫描点云。" }
    ],
    strip: ["points", "labels", "feats", "normals", "pavement_type"],
    notes: "这里要讲清楚点云生成是项目的重要部分。导师如果问细节，可以展开 ISO 8608、病害原语和 LiDAR 重采样。"
  },
  {
    label: "05",
    section: "训练策略",
    type: "training",
    title: "训练先解决“能不能识别病害”，再解决“属于哪一类病害”",
    lead: "当前最稳的路线不是直接硬训 38 类，而是先做二分类稳定化，再把 backbone 迁移到更细粒度的类别体系。",
    steps: [
      { h: "二分类", b: "背景 / 病害" },
      { h: "阈值校准", b: "约 0.44 附近更优" },
      { h: "迁移骨干", b: "复用已学到的前景边界" },
      { h: "38 类细分", b: "分阶段逐步扩展" }
    ],
    details: [
      { h: "模型结构", b: "Swin3D 是当前更稳的精度基线；PointMamba-inspired 是轻量化候选路线。mHC 默认作为通道混合模块保留。" },
      { h: "训练配方", b: "Focal 抑制背景主导，Dice 稳定小目标，Edge 强化裂缝和边界。Muon + AdamW 用于混合优化，AdamW 可回退。" }
    ],
    notes: "这页要帮助导师理解为什么现在先做二分类。不是逃避 38 类，而是给 38 类找更稳的入口。"
  },
  {
    label: "06",
    section: "阶段性证据",
    type: "results",
    title: "当前结果能证明路线可行，但还不能被包装成最终泛化结论",
    lead: "最可靠的阶段性结论是：合成点云可以支撑模型学习病害前景，二分类路线已经从“能不能学”进入“如何提稳”。",
    metrics: [
      { v: "0.7319", l: "Swin3D + mHC\n校准后 global mIoU" },
      { v: "0.7052", l: "disease IoU" },
      { v: "0.9039", l: "precision" },
      { v: "0.7624", l: "recall" },
      { v: "0.5671", l: "PointMamba + mHC\n二分类短跑 mIoU" }
    ],
    explain: [
      { h: "可以说", b: "二分类训练路线有效，Swin3D 当前更稳，PointMamba 是值得继续推进的轻量候选。" },
      { h: "不能说满", b: "这些结果主要来自合成小型验证集，部分快速诊断验证 batch 很少，不能直接等同于大规模泛化。" }
    ],
    notes: "这一页要主动诚实，别把小数据结果说成最终结论。导师会更相信这种表达。"
  },
  {
    label: "07",
    section: "扩展支线",
    type: "gan",
    title: "GAN 和 end2end 不是当前主结果，而是后续走向真实域的接口",
    lead: "代码里的 gan_enhanced / end2end 需要讲，但要讲准定位：它们是合成到真实风格适配的支线，不是现在最可靠的主结论来源。",
    flow: ["合成点云", "StyleTransferGen", "WGAN 判别器", "分割模型"],
    bullets: [
      "生成器对坐标和法向量做风格残差调整。",
      "判别器判断点云风格是否更接近目标域。",
      "end2end 把风格调整与分割训练放入同一循环。",
      "主线稳定后，再用它验证合成到真实的域适配能力。"
    ],
    notes: "这页是为了避免导师看到代码后疑惑：为什么还有 GAN。要说这是未来路线，不是当前已经成熟的结论。"
  },
  {
    label: "08",
    section: "当前限制",
    type: "risk",
    title: "现在最该补的不是继续堆模块，而是扩大验证和统一口径",
    lead: "当前项目已经跑通主线，但要往论文或比赛成果推进，必须先解决可信度问题。",
    risks: [
      { h: "数据规模", b: "小型合成验证集容易让模型在少数 batch 上过拟合。" },
      { h: "评估口径", b: "batch mean mIoU 与全验证集 global mIoU 需要统一记录。" },
      { h: "38 类迁移", b: "从二分类直接跳到 38 类过陡，需要中间类别集。" },
      { h: "真实域差异", b: "真实点云加载、噪声分布和域适配还没有形成正式闭环。" }
    ],
    notes: "导师汇报里主动讲限制很重要。不是项目不行，而是说明你知道下一步该验证什么。"
  },
  {
    label: "09",
    section: "下一步希望导师给建议",
    type: "next",
    title: "下一阶段建议围绕“扩数据、稳基线、分阶段多类”推进",
    lead: "这页可以作为和导师讨论的落点：不是问泛泛的“下一步做什么”，而是请导师确认推进顺序和成果包装方向。",
    actions: [
      { h: "1. 扩大合成数据", b: "目标约 5000 个场景，优先保证可恢复生成、分批执行和统一元数据。" },
      { h: "2. 重做正式验证", b: "固定 Swin3D + mHC 二分类基线，统一 global mIoU、disease IoU、precision、recall。" },
      { h: "3. 分阶段回到 38 类", b: "二分类 -> 4/8 类或高频子类 -> 38 类，避免冷启动直接塌缩。" },
      { h: "4. 再接真实域", b: "主线稳定后，再接 GAN / end2end 和真实点云验证。" }
    ],
    ask: "建议当面请导师判断：这个项目优先按论文路线推进，还是先按比赛展示路线包装？",
    notes: "收尾要提出一个可讨论的问题，让导师能给方向。"
  }
];

function esc(s) {
  return String(s).replace(/[&<>]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[ch]));
}

function renderSlide(s, i) {
  const num = s.label;
  if (s.type === "cover") {
    return `<section class="slide cover" data-slide="${i}">
      <div class="small-label">${esc(s.section)}</div>
      <div class="cover-grid">
        <div>
          <h1>${esc(s.title)}</h1>
          <p class="subtitle">${esc(s.subtitle)}</p>
          <div class="rule"></div>
          <p class="thesis">${esc(s.thesis)}</p>
        </div>
        <div class="agenda">
          <div class="agenda-title">汇报要回答的四个问题</div>
          ${s.points.map((p, idx) => `<div class="agenda-row"><span>${String(idx + 1).padStart(2, "0")}</span>${esc(p)}</div>`).join("")}
        </div>
      </div>
      <div class="page">${num}</div>
    </section>`;
  }
  if (s.type === "problem") {
    return `<section class="slide" data-slide="${i}">
      ${header(s)}<p class="lead wide">${esc(s.lead)}</p>
      <div class="four-grid problem-grid">${s.blocks.map(b => card(b.k, b.t)).join("")}</div>
      <div class="bottom-line"><strong>讲法：</strong>${esc(s.bottom)}</div>
      <div class="page">${num}</div>
    </section>`;
  }
  if (s.type === "loop") {
    return `<section class="slide" data-slide="${i}">
      ${header(s)}<p class="lead wide">${esc(s.lead)}</p>
      <div class="loop-row">${s.nodes.map((n, idx) => `<div class="loop-node"><span>${String(idx + 1).padStart(2, "0")}</span><h3>${esc(n.h)}</h3><p>${esc(n.b)}</p></div>${idx < s.nodes.length - 1 ? `<div class="arrow">→</div>` : ""}`).join("")}</div>
      <div class="quote-note">${esc(s.callout)}</div>
      <div class="page">${num}</div>
    </section>`;
  }
  if (s.type === "threeLayer") {
    return `<section class="slide" data-slide="${i}">
      ${header(s)}<p class="lead wide">${esc(s.lead)}</p>
      <div class="layer-stack">${s.layers.map((l, idx) => `<div class="layer"><div class="layer-num">${String(idx + 1).padStart(2, "0")}</div><div><h3>${esc(l.h)}</h3><p>${esc(l.b)}</p></div></div>`).join("")}</div>
      <div class="npz-strip"><span>.npz 输出</span>${s.strip.map(x => `<b>${esc(x)}</b>`).join("")}</div>
      <div class="page">${num}</div>
    </section>`;
  }
  if (s.type === "training") {
    return `<section class="slide" data-slide="${i}">
      ${header(s)}<p class="lead wide">${esc(s.lead)}</p>
      <div class="steps-row">${s.steps.map((st, idx) => `<div class="step"><span>${String(idx + 1).padStart(2, "0")}</span><h3>${esc(st.h)}</h3><p>${esc(st.b)}</p></div>${idx < s.steps.length - 1 ? `<div class="thin-arrow">→</div>` : ""}`).join("")}</div>
      <div class="two-notes">${s.details.map(d => card(d.h, d.b)).join("")}</div>
      <div class="page">${num}</div>
    </section>`;
  }
  if (s.type === "results") {
    return `<section class="slide" data-slide="${i}">
      ${header(s)}<p class="lead wide">${esc(s.lead)}</p>
      <div class="metrics">${s.metrics.map((m, idx) => `<div class="metric ${idx === 4 ? "secondary" : ""}"><div>${esc(m.v)}</div><p>${esc(m.l).replace(/\n/g, "<br>")}</p></div>`).join("")}</div>
      <div class="two-notes evidence">${s.explain.map(d => card(d.h, d.b)).join("")}</div>
      <div class="page">${num}</div>
    </section>`;
  }
  if (s.type === "gan") {
    return `<section class="slide" data-slide="${i}">
      ${header(s)}<p class="lead wide">${esc(s.lead)}</p>
      <div class="gan-layout"><div class="gan-flow">
        <div class="flow-item flow-a"><span>01</span><h3>${esc(s.flow[0])}</h3></div>
        <div class="flow-arrow flow-ab">→</div>
        <div class="flow-item flow-b"><span>02</span><h3>${esc(s.flow[1])}</h3></div>
        <div class="flow-arrow flow-bc">→</div>
        <div class="flow-item flow-c"><span>03</span><h3>${esc(s.flow[2])}</h3></div>
        <div class="flow-arrow flow-bd">↓</div>
        <div class="flow-item flow-d"><span>04</span><h3>${esc(s.flow[3])}</h3></div>
      </div>
      <div class="bullet-panel">${s.bullets.map(x => `<p>${esc(x)}</p>`).join("")}</div></div>
      <div class="page">${num}</div>
    </section>`;
  }
  if (s.type === "risk") {
    return `<section class="slide" data-slide="${i}">
      ${header(s)}<p class="lead wide">${esc(s.lead)}</p>
      <div class="risk-grid">${s.risks.map((r, idx) => `<div class="risk"><span>${String(idx + 1).padStart(2, "0")}</span><h3>${esc(r.h)}</h3><p>${esc(r.b)}</p></div>`).join("")}</div>
      <div class="page">${num}</div>
    </section>`;
  }
  if (s.type === "next") {
    return `<section class="slide" data-slide="${i}">
      ${header(s)}<p class="lead wide">${esc(s.lead)}</p>
      <div class="actions">${s.actions.map(a => `<div class="action"><h3>${esc(a.h)}</h3><p>${esc(a.b)}</p></div>`).join("")}</div>
      <div class="ask">${esc(s.ask)}</div>
      <div class="page">${num}</div>
    </section>`;
  }
}

function header(s) {
  return `<div class="small-label">${esc(s.section)}</div><h2>${esc(s.title)}</h2>`;
}

function card(h, b) {
  return `<div class="card"><h3>${esc(h)}</h3><p>${esc(b)}</p></div>`;
}

const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>RoadMC 导师汇报</title>
<style>
:root {
  --paper: #f7f0e6;
  --paper2: #fbf7ef;
  --ink: #2c2118;
  --muted: #6f6258;
  --soft: #e9dacb;
  --line: #d8c2af;
  --accent: #c76543;
  --accent2: #e4a27e;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: #e9e1d7; color: var(--ink); font-family: "DengXian", "Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }
.deck { width: 1920px; min-height: 1080px; }
.slide { position: relative; width: 1920px; height: 1080px; overflow: hidden; background: radial-gradient(circle at 15% 12%, #fffaf3 0, var(--paper) 44%, #f4eadf 100%); padding: 72px 96px; }
.slide::before { content: ""; position: absolute; inset: 36px; border: 1px solid rgba(216, 194, 175, .32); pointer-events: none; }
.small-label { color: var(--accent); font-size: 25px; font-weight: 600; margin-bottom: 42px; }
h1, h2, h3, p { margin: 0; letter-spacing: 0; }
h1 { font-size: 98px; font-weight: 600; line-height: 1; }
h2 { width: 1420px; font-size: 54px; font-weight: 600; line-height: 1.22; text-wrap: pretty; }
h3 { font-size: 30px; font-weight: 600; line-height: 1.3; }
p { color: var(--muted); font-size: 25px; line-height: 1.48; text-wrap: pretty; }
.lead { margin-top: 26px; font-size: 29px; line-height: 1.55; }
.wide { width: 1320px; }
.page { position: absolute; right: 94px; bottom: 70px; color: #8d7c6f; font-size: 18px; }
.rule { width: 260px; height: 2px; background: var(--line); margin: 48px 0 0; }
.cover-grid { display: grid; grid-template-columns: 1fr 460px; gap: 130px; align-items: start; margin-top: 46px; }
.subtitle { margin-top: 30px; font-size: 36px; color: var(--muted); }
.thesis { margin-top: 250px; width: 980px; color: var(--ink); font-size: 37px; line-height: 1.46; }
.agenda { background: rgba(239, 226, 211, .86); border: 1px solid var(--line); border-radius: 18px; padding: 44px 44px 34px; margin-top: 20px; }
.agenda-title { font-size: 32px; font-weight: 620; margin-bottom: 34px; }
.agenda-row { display: grid; grid-template-columns: 56px 1fr; gap: 18px; padding: 22px 0; border-top: 1px solid rgba(216, 194, 175, .65); font-size: 27px; }
.agenda-row span, .loop-node span, .step span, .metric div, .risk span, .layer-num, .flow-item span { color: var(--accent); font-weight: 650; }
.four-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 30px; width: 1260px; margin-top: 60px; }
.card { background: rgba(251, 247, 239, .82); border: 1px solid var(--line); border-radius: 18px; padding: 34px 38px; min-height: 164px; }
.card p { margin-top: 18px; font-size: 24px; }
.bottom-line, .quote-note, .ask { position: absolute; left: 96px; right: 96px; bottom: 134px; background: rgba(239, 226, 211, .7); border: 1px solid var(--line); border-radius: 14px; padding: 26px 36px; font-size: 28px; line-height: 1.4; }
.bottom-line strong { color: var(--accent); margin-right: 22px; }
.loop-row { display: grid; grid-template-columns: 250px 48px 250px 48px 250px 48px 250px 48px 250px; align-items: center; margin-top: 74px; }
.loop-node, .step, .flow-item { background: rgba(251, 247, 239, .8); border: 1px solid var(--line); border-radius: 18px; padding: 32px 34px; min-height: 190px; }
.loop-node h3, .step h3, .flow-item h3 { margin-top: 26px; }
.loop-node p, .step p { margin-top: 16px; font-size: 23px; }
.arrow, .thin-arrow, .flow-arrow { text-align: center; color: var(--accent2); font-size: 38px; }
.layer-stack { display: grid; grid-template-columns: repeat(3, 1fr); gap: 32px; margin-top: 70px; width: 1450px; }
.layer { min-height: 330px; display: grid; grid-template-columns: 64px 1fr; gap: 24px; background: rgba(251, 247, 239, .82); border: 1px solid var(--line); border-radius: 20px; padding: 42px; }
.layer p { margin-top: 24px; font-size: 25px; color: var(--muted); }
.npz-strip { position: absolute; left: 160px; right: 160px; bottom: 160px; display: flex; align-items: center; gap: 28px; background: rgba(239, 226, 211, .72); border: 1px solid var(--line); border-radius: 14px; padding: 28px 36px; }
.npz-strip span { color: var(--accent); font-size: 30px; font-weight: 650; margin-right: 20px; }
.npz-strip b { font-size: 25px; font-weight: 520; padding: 8px 18px; border: 1px solid rgba(216, 194, 175, .7); border-radius: 999px; background: var(--paper2); }
.steps-row { display: grid; grid-template-columns: 280px 52px 280px 52px 280px 52px 280px; align-items: center; margin-top: 70px; margin-left: 40px; }
.step { min-height: 176px; }
.two-notes { display: grid; grid-template-columns: 1fr 1fr; gap: 70px; width: 1510px; margin-top: 76px; }
.two-notes .card { min-height: 188px; border-left: 8px solid var(--accent); }
.metrics { display: grid; grid-template-columns: 1.1fr 1fr 1fr 1fr 1.05fr; gap: 26px; width: 1600px; margin-top: 64px; }
.metric { background: rgba(251, 247, 239, .82); border: 1px solid var(--line); border-radius: 18px; padding: 36px 42px; min-height: 210px; }
.metric div { font-size: 54px; line-height: 1; }
.metric.secondary div { color: var(--ink); }
.metric p { margin-top: 34px; font-size: 23px; }
.evidence { margin-top: 70px; }
.gan-layout { display: grid; grid-template-columns: 1fr 510px; gap: 64px; align-items: start; margin-top: 70px; }
.gan-flow { display: grid; grid-template-columns: 245px 48px 300px 48px 245px; grid-template-rows: 180px 70px 180px; gap: 0; align-items: center; }
.flow-a { grid-column: 1; grid-row: 1; }
.flow-ab { grid-column: 2; grid-row: 1; }
.flow-b { grid-column: 3; grid-row: 1; }
.flow-bc { grid-column: 4; grid-row: 1; }
.flow-c { grid-column: 5; grid-row: 1; }
.flow-bd { grid-column: 3; grid-row: 2; align-self: center; }
.flow-d { grid-column: 3; grid-row: 3; }
.flow-item { min-height: 170px; }
.flow-item h3 { font-size: 28px; }
.bullet-panel { background: rgba(239, 226, 211, .72); border: 1px solid var(--line); border-radius: 18px; padding: 34px 42px; }
.bullet-panel p { color: var(--ink); font-size: 25px; padding: 18px 0; border-bottom: 1px solid rgba(216, 194, 175, .62); }
.bullet-panel p:last-child { border-bottom: 0; }
.risk-grid { width: 1480px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 30px; margin-top: 82px; }
.risk { background: rgba(251, 247, 239, .82); border: 1px solid var(--line); border-radius: 18px; min-height: 390px; padding: 42px 38px; }
.risk h3 { margin-top: 54px; }
.risk p { margin-top: 28px; font-size: 25px; }
.actions { display: grid; grid-template-columns: repeat(2, 1fr); gap: 26px 38px; width: 1500px; margin-top: 58px; }
.action { background: rgba(251, 247, 239, .82); border: 1px solid var(--line); border-radius: 18px; padding: 32px 38px; min-height: 150px; }
.action p { margin-top: 14px; font-size: 24px; }
.ask { bottom: 130px; font-size: 30px; color: var(--ink); border-left: 8px solid var(--accent); }
@media print { .slide { page-break-after: always; } }
</style>
</head>
<body>
<div class="deck">
${slides.map(renderSlide).join("\n")}
</div>
<script>
const params = new URLSearchParams(location.search);
const target = params.get('slide');
if (target) {
  document.querySelectorAll('.slide').forEach((el, idx) => {
    el.style.display = String(idx) === target ? 'block' : 'none';
  });
}
</script>
</body>
</html>`;

await fs.mkdir(outDir, { recursive: true });
await fs.mkdir(pngDir, { recursive: true });
await fs.writeFile(htmlPath, html, "utf8");

const fileUrl = pathToFileURL(htmlPath).href;
for (let i = 0; i < slides.length; i++) {
  const screenshotPath = path.join(pngDir, `slide-${String(i + 1).padStart(2, "0")}.png`);
  const result = spawnSync(chromePath, [
    "--headless=new",
    "--disable-gpu",
    "--hide-scrollbars",
    "--force-device-scale-factor=1",
    "--window-size=1920,1080",
    `--screenshot=${screenshotPath}`,
    `${fileUrl}?slide=${i}`,
  ], { encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(`Chrome screenshot failed for slide ${i + 1}:\n${result.stderr || result.stdout}`);
  }
}

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.subject = "RoadMC 导师汇报";
pptx.title = "RoadMC 导师汇报";
pptx.company = "RoadMC";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Microsoft YaHei UI",
  bodyFontFace: "Microsoft YaHei UI",
  lang: "zh-CN"
};
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";
for (let i = 0; i < slides.length; i++) {
  const slide = pptx.addSlide();
  slide.background = { color: "F7F0E6" };
  slide.addImage({ path: path.join(pngDir, `slide-${String(i + 1).padStart(2, "0")}.png`), x: 0, y: 0, w: 13.333333, h: 7.5 });
  if (typeof slide.addNotes === "function") slide.addNotes(slides[i].notes || "");
}
await pptx.writeFile({ fileName: pptxPath });

console.log(JSON.stringify({ htmlPath, pptxPath, pngDir, slideCount: slides.length }, null, 2));





