import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";

const require = createRequire(import.meta.url);
const pptxgen = require("C:/Users/SEELE/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = "C:/Users/SEELE/PycharmProjects/PythonProject";
const outDir = path.join(root, "output/ppt/roadmc_advisor_design_v2");
const htmlPath = path.join(outDir, "roadmc_advisor_design_v2.html");
const pngDir = path.join(outDir, "slides_png");
const pptxPath = path.join(root, "output/ppt/roadmc_advisor_for_teacher_design_v2.pptx");
const chromePath = "C:/Program Files/Google/Chrome/Application/chrome.exe";

const notes = [
  "开场先讲定位：RoadMC 不是一个单独分割模型，而是围绕路面病害点云任务建立“生成、训练、评估”的闭环。",
  "这一页只做问题框定。重点是说明真实点云监督成本、病害稀疏和长尾类别会直接决定路线设计。",
  "这页把生成模块提升为方法本体。强调生成不是临时造数据，而是控制几何、标签和监督上限。",
  "讲清楚点云如何生成：路面形貌、病害原语、传感器观测，最后输出可训练的 npz 样本。",
  "解释为什么先做二分类。它不是放弃 38 类，而是先稳定前景边界，再迁移到更细粒度类别。",
  "主动讲证据边界。当前结果能证明路线可行，但仍基于合成小型测试集，不能包装成最终泛化结论。",
  "GAN 和 end2end 要定位为未来真实域适配接口，不要把它们说成当前主结果。",
  "这页主动暴露风险，便于导师判断下一步验证优先级。",
  "收尾用两个路线选择引导讨论：论文路线与展示路线。请导师判断优先推进哪一种。"
];

const html = String.raw`<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>RoadMC 导师汇报设计版</title>
<style>
@font-face {
  font-family: "RoadMC Styrene";
  src: url("file:///C:/Users/SEELE/Documents/test/fonts/Styrene/StyreneAWeb-Bold.ttf") format("truetype");
  font-weight: 700;
}
@font-face {
  font-family: "RoadMC Styrene";
  src: url("file:///C:/Users/SEELE/Documents/test/fonts/Styrene/StyreneAWeb-Light.ttf") format("truetype");
  font-weight: 300;
}
@font-face {
  font-family: "RoadMC Tiempos";
  src: url("file:///C:/Users/SEELE/Documents/test/fonts/Tiempos/TiemposText-Regular.ttf") format("truetype");
  font-weight: 400;
}
@font-face {
  font-family: "RoadMC Tiempos";
  src: url("file:///C:/Users/SEELE/Documents/test/fonts/Tiempos/TiemposText-Semibold.ttf") format("truetype");
  font-weight: 650;
}
@font-face {
  font-family: "RoadMC Serif SC";
  src: url("file:///C:/Windows/Fonts/NotoSerifSC-VF.ttf") format("truetype");
}
@font-face {
  font-family: "RoadMC Sans SC";
  src: url("file:///C:/Windows/Fonts/NotoSansSC-VF.ttf") format("truetype");
}

:root {
  --paper: #f5eadc;
  --paper-light: #fbf6ee;
  --paper-strong: #ead8c4;
  --ink: #211915;
  --text: #3b2c24;
  --muted: #78675b;
  --line: #c8ad96;
  --line-soft: rgba(200, 173, 150, 0.48);
  --accent: #a9472c;
  --accent-soft: #dc8b67;
  --green: #526d61;
  --blue: #465f76;
  --grid: rgba(111, 88, 72, 0.055);
}

* { box-sizing: border-box; }
html, body {
  margin: 0;
  background: #d8ccbf;
  color: var(--ink);
  font-family: "RoadMC Sans SC", "Noto Sans SC", "Microsoft YaHei UI", sans-serif;
  text-rendering: geometricPrecision;
}
.deck { width: 1920px; min-height: 1080px; }
.slide {
  position: relative;
  width: 1920px;
  height: 1080px;
  overflow: hidden;
  padding: 74px 96px 78px 112px;
  background:
    linear-gradient(90deg, var(--grid) 1px, transparent 1px) 0 0 / 48px 48px,
    linear-gradient(0deg, var(--grid) 1px, transparent 1px) 0 0 / 48px 48px,
    linear-gradient(145deg, #fbf5ea 0%, var(--paper) 57%, #ecddca 100%);
}
.slide::after {
  content: "";
  position: absolute;
  left: 40px;
  right: 40px;
  top: 36px;
  bottom: 36px;
  border: 1px solid rgba(160, 130, 105, 0.22);
  pointer-events: none;
}
h1, h2, h3, p, div { margin: 0; letter-spacing: 0; }
h1, h2 {
  font-family: "RoadMC Serif SC", "Noto Serif SC", "SimSun", serif;
  font-weight: 650;
  line-height: 1.08;
  text-wrap: pretty;
}
h1 { font-size: 118px; }
h2 { max-width: 1280px; font-size: 63px; }
h3 {
  font-size: 30px;
  line-height: 1.25;
  font-weight: 700;
}
p {
  color: var(--muted);
  font-size: 26px;
  line-height: 1.55;
  text-wrap: pretty;
}
.kicker {
  font-family: "RoadMC Styrene", "RoadMC Sans SC", sans-serif;
  color: var(--accent);
  font-size: 18px;
  font-weight: 700;
  text-transform: uppercase;
}
.section-label {
  display: grid;
  grid-template-columns: 68px 1fr;
  gap: 18px;
  align-items: center;
  margin-bottom: 42px;
}
.section-label .num {
  font-family: "RoadMC Styrene", sans-serif;
  font-size: 22px;
  color: var(--accent);
}
.section-label .line {
  width: 560px;
  height: 1px;
  background: var(--line-soft);
}
.lead {
  margin-top: 26px;
  max-width: 1160px;
  font-size: 30px;
}
.folio {
  position: absolute;
  right: 96px;
  bottom: 64px;
  z-index: 3;
  display: flex;
  gap: 18px;
  align-items: center;
  font-family: "RoadMC Styrene", sans-serif;
  color: #806c5f;
  font-size: 16px;
}
.folio::before { content: ""; width: 86px; height: 1px; background: var(--line); }
.hairline { height: 1px; background: var(--line-soft); }
.tag {
  display: inline-flex;
  height: 34px;
  padding: 0 14px;
  align-items: center;
  border: 1px solid var(--line);
  color: var(--accent);
  font-family: "RoadMC Styrene", "RoadMC Sans SC", sans-serif;
  font-size: 15px;
  font-weight: 700;
  background: rgba(251, 246, 238, 0.62);
}
.paper-panel {
  background: rgba(251, 246, 238, 0.74);
  border: 1px solid var(--line-soft);
}
.cover {
  padding-top: 86px;
}
.cover > div:first-child {
  position: absolute;
  left: 112px;
  top: 86px;
  width: 900px;
}
.cover-mark {
  font-family: "RoadMC Styrene", sans-serif;
  font-size: 19px;
  color: var(--accent);
  margin-bottom: 124px;
}
.cover h1 {
  font-family: "RoadMC Styrene", "RoadMC Serif SC", sans-serif;
  font-size: 148px;
  line-height: 0.92;
  letter-spacing: 0;
}
.cover-sub {
  margin-top: 22px;
  max-width: 840px;
  font-family: "RoadMC Serif SC", serif;
  color: var(--text);
  font-size: 42px;
  line-height: 1.26;
}
.cover-thesis {
  position: absolute;
  left: 112px;
  bottom: 142px;
  width: 920px;
  color: var(--ink);
  font-size: 32px;
  line-height: 1.48;
}
.cover-ledger {
  position: absolute;
  right: 112px;
  top: 154px;
  width: 570px;
  margin-top: 0;
  border-top: 2px solid var(--ink);
}
.ledger-title {
  padding: 20px 0 24px;
  font-family: "RoadMC Serif SC", serif;
  font-size: 29px;
}
.ledger-row {
  display: grid;
  grid-template-columns: 76px 1fr;
  border-top: 1px solid var(--line);
  padding: 23px 0;
  align-items: baseline;
}
.ledger-row span {
  font-family: "RoadMC Styrene", sans-serif;
  color: var(--accent);
  font-size: 22px;
}
.ledger-row b {
  font-size: 29px;
  font-weight: 650;
}
.cover-sig {
  position: absolute;
  right: 96px;
  bottom: 142px;
  width: 520px;
  padding-top: 18px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-family: "RoadMC Tiempos", serif;
  font-size: 24px;
  line-height: 1.35;
}
.problem-map {
  margin-top: 82px;
  width: 1590px;
  display: grid;
  grid-template-columns: 330px 1fr 330px 1fr 330px;
  align-items: stretch;
}
.problem-node {
  min-height: 330px;
  padding: 30px 0 0;
  border-top: 2px solid var(--ink);
}
.problem-node .index, .matrix-cell .index, .risk-column .index {
  font-family: "RoadMC Styrene", sans-serif;
  color: var(--accent);
  font-size: 20px;
  font-weight: 700;
}
.problem-node h3 {
  margin-top: 48px;
  font-family: "RoadMC Serif SC", serif;
  font-size: 39px;
  line-height: 1.18;
}
.problem-node p { margin-top: 24px; max-width: 286px; }
.problem-link {
  position: relative;
  min-height: 330px;
}
.problem-link::before {
  content: "";
  position: absolute;
  top: 72px;
  left: 28px;
  right: 28px;
  height: 1px;
  background: var(--line);
}
.problem-link::after {
  content: "";
  position: absolute;
  top: 66px;
  right: 24px;
  width: 12px;
  height: 12px;
  border-right: 1px solid var(--line);
  border-top: 1px solid var(--line);
  transform: rotate(45deg);
}
.bottom-thesis {
  position: absolute;
  left: 112px;
  right: 264px;
  bottom: 142px;
  display: grid;
  grid-template-columns: 138px 1fr;
  gap: 34px;
  align-items: center;
  padding-top: 28px;
  border-top: 1px solid var(--line);
}
.bottom-thesis b {
  color: var(--accent);
  font-size: 27px;
}
.bottom-thesis p { color: var(--ink); font-size: 30px; }
.system-layout {
  margin-top: 56px;
  display: grid;
  grid-template-columns: 640px 1fr;
  gap: 88px;
  align-items: center;
}
.system-statement {
  height: 486px;
  padding: 48px 50px;
  border-left: 8px solid var(--accent);
  background: rgba(251, 246, 238, 0.56);
}
.system-statement h3 {
  font-family: "RoadMC Serif SC", serif;
  font-size: 52px;
  line-height: 1.15;
}
.system-statement p {
  margin-top: 34px;
  color: var(--text);
  font-size: 29px;
}
.loop-canvas {
  position: relative;
  width: 900px;
  height: 560px;
}
.loop-ring {
  position: absolute;
  left: 188px;
  top: 72px;
  width: 455px;
  height: 455px;
  border: 1px solid var(--line);
}
.loop-ring::before, .loop-ring::after {
  content: "";
  position: absolute;
  inset: 44px;
  border: 1px solid rgba(169, 71, 44, 0.32);
}
.loop-ring::after {
  inset: 118px;
  background: rgba(251, 246, 238, 0.74);
}
.loop-center {
  position: absolute;
  left: 306px;
  top: 238px;
  width: 220px;
  text-align: center;
  z-index: 2;
  font-family: "RoadMC Serif SC", serif;
  font-size: 29px;
  line-height: 1.26;
}
.loop-item {
  position: absolute;
  width: 258px;
  padding: 22px 0 18px;
  border-top: 2px solid var(--ink);
  background: rgba(245, 234, 220, 0.76);
}
.loop-item span {
  font-family: "RoadMC Styrene", sans-serif;
  color: var(--accent);
  font-size: 16px;
  font-weight: 700;
}
.loop-item h3 { margin-top: 17px; font-size: 29px; }
.loop-item p { margin-top: 10px; font-size: 22px; }
.li-1 { left: 0; top: 38px; }
.li-2 { right: 10px; top: 38px; }
.li-3 { right: 0; bottom: 34px; }
.li-4 { left: 0; bottom: 34px; }
.synth-layout {
  margin-top: 38px;
  display: grid;
  grid-template-columns: 1fr 540px;
  gap: 76px;
}
.pipeline {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
}
.pipeline-row {
  display: grid;
  grid-template-columns: 106px 360px 1fr;
  gap: 34px;
  align-items: center;
  min-height: 134px;
  border-top: 1px solid var(--line);
  padding-top: 16px;
}
.pipeline-row .step-no {
  font-family: "RoadMC Styrene", sans-serif;
  color: var(--accent);
  font-size: 36px;
  font-weight: 700;
}
.pipeline-row h3 {
  font-family: "RoadMC Serif SC", serif;
  font-size: 36px;
}
.pipeline-row p { font-size: 23px; }
.mini-plot {
  position: relative;
  height: 350px;
  border: 1px solid var(--line-soft);
  background:
    linear-gradient(90deg, rgba(169, 71, 44, 0.08) 1px, transparent 1px) 0 0 / 36px 36px,
    rgba(251, 246, 238, 0.62);
}
.mini-plot svg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}
.npz-output {
  margin-top: 24px;
  border-top: 2px solid var(--ink);
  padding-top: 18px;
}
.npz-output h3 {
  font-family: "RoadMC Styrene", sans-serif;
  color: var(--accent);
  font-size: 23px;
}
.npz-list {
  margin-top: 14px;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 6px 22px;
}
.npz-list span {
  border-bottom: 1px solid var(--line-soft);
  padding: 9px 0;
  font-family: "RoadMC Tiempos", "RoadMC Sans SC", serif;
  font-size: 22px;
}
.train-map {
  margin-top: 62px;
  display: grid;
  grid-template-columns: 390px 1fr;
  gap: 70px;
}
.stages {
  border-top: 2px solid var(--ink);
}
.stage {
  display: grid;
  grid-template-columns: 70px 1fr;
  gap: 20px;
  padding: 26px 0;
  border-bottom: 1px solid var(--line-soft);
}
.stage span {
  font-family: "RoadMC Styrene", sans-serif;
  color: var(--accent);
  font-size: 20px;
  font-weight: 700;
}
.stage h3 { font-size: 32px; }
.stage p { margin-top: 8px; font-size: 23px; }
.architecture {
  position: relative;
  min-height: 492px;
}
.arch-row {
  display: grid;
  grid-template-columns: 200px 64px 230px 64px 190px 64px 230px;
  align-items: center;
  margin-top: 28px;
}
.arch-box {
  height: 132px;
  border: 1px solid var(--line);
  background: rgba(251, 246, 238, 0.72);
  padding: 24px;
}
.arch-box h3 {
  font-size: 26px;
  font-family: "RoadMC Serif SC", serif;
}
.arch-box p { margin-top: 10px; font-size: 20px; }
.arch-arrow {
  color: var(--accent-soft);
  font-family: "RoadMC Styrene", sans-serif;
  font-size: 36px;
  text-align: center;
}
.loss-band {
  margin-top: 44px;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 18px;
}
.loss-item {
  padding: 20px 22px;
  border-top: 2px solid var(--accent);
  background: rgba(251, 246, 238, 0.56);
}
.loss-item b {
  font-family: "RoadMC Styrene", sans-serif;
  font-size: 19px;
  color: var(--accent);
}
.loss-item p { margin-top: 11px; font-size: 20px; }
.evidence-layout {
  margin-top: 60px;
  display: grid;
  grid-template-columns: 1fr 500px;
  gap: 80px;
}
.metric-wall {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
}
.metric-large {
  grid-column: span 2;
  min-height: 210px;
  padding: 30px 0;
  border-top: 3px solid var(--accent);
  border-bottom: 1px solid var(--line);
}
.metric-small {
  min-height: 210px;
  padding: 30px 0;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}
.metric-value {
  font-family: "RoadMC Styrene", "RoadMC Tiempos", sans-serif;
  color: var(--accent);
  font-size: 75px;
  font-weight: 700;
  line-height: 0.94;
}
.metric-small .metric-value { color: var(--ink); font-size: 56px; }
.metric-caption {
  margin-top: 28px;
  color: var(--muted);
  font-size: 22px;
  line-height: 1.35;
}
.evidence-note {
  border-left: 1px solid var(--line);
  padding-left: 36px;
}
.evidence-note h3 {
  font-family: "RoadMC Serif SC", serif;
  font-size: 44px;
  line-height: 1.2;
}
.evidence-note p {
  margin-top: 26px;
  color: var(--text);
  font-size: 27px;
}
.caveat {
  margin-top: 34px;
  padding-top: 26px;
  border-top: 1px solid var(--line);
  color: var(--accent);
  font-size: 25px;
}
.gan-map {
  margin-top: 58px;
  display: grid;
  grid-template-columns: 980px 510px;
  gap: 86px;
}
.main-branch, .future-branch {
  border-top: 2px solid var(--ink);
  padding-top: 28px;
}
.branch-title {
  font-family: "RoadMC Styrene", sans-serif;
  color: var(--accent);
  font-size: 19px;
  font-weight: 700;
}
.branch-flow {
  margin-top: 42px;
  display: grid;
  grid-template-columns: 210px 54px 250px 54px 210px;
  align-items: center;
}
.branch-box {
  min-height: 126px;
  border: 1px solid var(--line);
  padding: 24px;
  background: rgba(251, 246, 238, 0.64);
}
.branch-box h3 { font-size: 26px; }
.branch-box p { margin-top: 11px; font-size: 20px; }
.branch-arrow {
  color: var(--accent-soft);
  text-align: center;
  font-family: "RoadMC Styrene", sans-serif;
  font-size: 34px;
}
.future-list {
  margin-top: 32px;
}
.future-list p {
  padding: 19px 0;
  border-bottom: 1px solid var(--line-soft);
  color: var(--text);
  font-size: 25px;
}
.risk-matrix {
  margin-top: 62px;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 34px;
  width: 1580px;
}
.risk-column {
  min-height: 430px;
  border-top: 2px solid var(--ink);
  padding-top: 26px;
}
.risk-column h3 {
  margin-top: 50px;
  font-family: "RoadMC Serif SC", serif;
  font-size: 37px;
}
.risk-column p {
  margin-top: 24px;
  font-size: 24px;
}
.risk-column .fix {
  margin-top: 40px;
  padding-top: 22px;
  border-top: 1px solid var(--line);
  color: var(--accent);
  font-size: 22px;
}
.next-layout {
  margin-top: 64px;
  display: grid;
  grid-template-columns: 1fr 560px;
  gap: 86px;
}
.roadmap {
  border-top: 2px solid var(--ink);
}
.roadmap-row {
  display: grid;
  grid-template-columns: 84px 300px 1fr;
  gap: 30px;
  padding: 27px 0;
  border-bottom: 1px solid var(--line-soft);
  align-items: start;
}
.roadmap-row span {
  font-family: "RoadMC Styrene", sans-serif;
  color: var(--accent);
  font-size: 22px;
  font-weight: 700;
}
.roadmap-row h3 { font-size: 31px; }
.roadmap-row p { font-size: 23px; }
.decision {
  min-height: 500px;
  padding: 38px 42px;
  background: rgba(251, 246, 238, 0.68);
  border: 1px solid var(--line);
}
.decision h3 {
  font-family: "RoadMC Serif SC", serif;
  font-size: 43px;
}
.decision p {
  margin-top: 28px;
  color: var(--text);
  font-size: 27px;
}
.choice {
  margin-top: 36px;
  display: grid;
  gap: 18px;
}
.choice div {
  padding: 18px 0;
  border-top: 1px solid var(--line);
  font-size: 25px;
}
.choice b { color: var(--accent); margin-right: 16px; }
@media print { .slide { page-break-after: always; } }
</style>
</head>
<body>
<main class="deck">
<section class="slide cover" data-screen-label="01 Cover">
  <div>
    <div class="cover-mark">MENTOR RESEARCH BRIEFING / 2026</div>
    <h1>RoadMC</h1>
    <div class="cover-sub">物理仿真点云生成与路面病害分割训练闭环</div>
  </div>
  <aside class="cover-ledger">
    <div class="ledger-title">这次汇报要回答的问题</div>
    <div class="ledger-row"><span>01</span><b>监督数据从哪里来</b></div>
    <div class="ledger-row"><span>02</span><b>模型路线怎样稳定</b></div>
    <div class="ledger-row"><span>03</span><b>结果边界如何解释</b></div>
    <div class="ledger-row"><span>04</span><b>下一步验证怎么做</b></div>
  </aside>
  <p class="cover-thesis">核心不是单个网络，而是一套从可控数据生成到病害分割训练、再到评估校准的研究闭环。</p>
  <div class="cover-sig">Synthesis is treated as a first-class method component, not a disposable preprocessing step.</div>
  <div class="folio">01</div>
</section>

<section class="slide" data-screen-label="02 Problem">
  <div class="section-label"><div class="num">02</div><div><div class="kicker">Problem Framing</div><div class="line"></div></div></div>
  <h2>直接训练路面病害点云模型，很容易卡在数据和长尾上</h2>
  <p class="lead">如果直接从真实点云训练 38 类病害分割，最早遇到的通常不是网络结构上限，而是监督数据、标签成本和类别分布共同造成的不稳定。</p>
  <div class="problem-map">
    <div class="problem-node"><div class="index">A</div><h3>真实采集贵</h3><p>点云采集、清洗和逐点标注成本高，早期很难快速形成可控监督。</p></div>
    <div class="problem-link"></div>
    <div class="problem-node"><div class="index">B</div><h3>病害目标稀疏</h3><p>裂缝、剥落、坑槽等点占比低，模型容易偏向预测背景。</p></div>
    <div class="problem-link"></div>
    <div class="problem-node"><div class="index">C</div><h3>38 类长尾明显</h3><p>少数类别样本不足时，冷启动多分类训练容易塌缩。</p></div>
  </div>
  <div class="bottom-thesis"><b>讲法</b><p>RoadMC 的价值不只是换网络，而是先把“可控监督”和“稳定训练路线”建立起来。</p></div>
  <div class="folio">02</div>
</section>

<section class="slide" data-screen-label="03 Method Positioning">
  <div class="section-label"><div class="num">03</div><div><div class="kicker">Method Positioning</div><div class="line"></div></div></div>
  <h2>把点云生成提升为方法的一半，再用训练结果反向校准生成</h2>
  <div class="system-layout">
    <div class="system-statement">
      <h3>生成不是临时造数据，而是在定义监督空间。</h3>
      <p>生成模块决定几何形态、标签分布、噪声结构和训练可见的病害边界；训练模块验证这些监督是否真的能被模型学到。</p>
    </div>
    <div class="loop-canvas">
      <div class="loop-ring"></div>
      <div class="loop-center">生成 / 训练 / 评估闭环</div>
      <div class="loop-item li-1"><span>01</span><h3>物理先验</h3><p>粗糙度、微纹理、路面形貌</p></div>
      <div class="loop-item li-2"><span>02</span><h3>病害几何</h3><p>裂缝、坑槽、车辙、剥落</p></div>
      <div class="loop-item li-3"><span>03</span><h3>分割训练</h3><p>Swin3D / mHC / binary first</p></div>
      <div class="loop-item li-4"><span>04</span><h3>评估校准</h3><p>mIoU、阈值、误差回看</p></div>
    </div>
  </div>
  <div class="folio">03</div>
</section>

<section class="slide" data-screen-label="04 Synthesis">
  <div class="section-label"><div class="num">04</div><div><div class="kicker">Point Cloud Synthesis</div><div class="line"></div></div></div>
  <h2>合成点云来自一条物理和几何流水线，不是随机造点</h2>
  <p class="lead">生成流程按“路面形貌、病害形变、传感器观测”组织，最后输出带逐点标签的训练样本。</p>
  <div class="synth-layout">
    <div class="pipeline">
      <div class="pipeline-row"><div class="step-no">01</div><h3>路面形貌</h3><p>用 ISO 8608 粗糙度谱生成宏观起伏，再叠加 fBm 微纹理和局部法向变化。</p></div>
      <div class="pipeline-row"><div class="step-no">02</div><h3>病害形变</h3><p>通过裂缝、坑槽、车辙、剥落等病害原语写入几何结构，标签从几何区域映射得到。</p></div>
      <div class="pipeline-row"><div class="step-no">03</div><h3>LiDAR 观测</h3><p>用扫描线重采样、距离噪声和点密度控制，让输入更接近真实扫描点云。</p></div>
    </div>
    <aside>
      <div class="mini-plot">
        <svg viewBox="0 0 540 420" aria-hidden="true">
          <path d="M36 250 C88 235 120 268 164 252 S242 222 298 244 S398 284 502 238" fill="none" stroke="#211915" stroke-width="3"/>
          <path d="M122 246 C128 208 145 199 156 170 C164 197 181 210 184 248" fill="none" stroke="#a9472c" stroke-width="4"/>
          <path d="M308 244 C332 210 370 214 390 242" fill="none" stroke="#a9472c" stroke-width="5"/>
          <g fill="#465f76" opacity="0.62">
            <circle cx="66" cy="276" r="4"/><circle cx="110" cy="270" r="4"/><circle cx="154" cy="282" r="4"/><circle cx="198" cy="260" r="4"/>
            <circle cx="242" cy="248" r="4"/><circle cx="286" cy="262" r="4"/><circle cx="330" cy="280" r="4"/><circle cx="374" cy="292" r="4"/>
            <circle cx="418" cy="274" r="4"/><circle cx="462" cy="260" r="4"/>
          </g>
          <path d="M58 104 L502 104" stroke="#c8ad96" stroke-width="1"/>
          <text x="58" y="88" fill="#a9472c" font-family="RoadMC Styrene" font-size="18">GEOMETRY + SENSOR OBSERVATION</text>
        </svg>
      </div>
      <div class="npz-output">
        <h3>.npz output</h3>
        <div class="npz-list"><span>points</span><span>labels</span><span>feats</span><span>normals</span><span>pavement_type</span></div>
      </div>
    </aside>
  </div>
  <div class="folio">04</div>
</section>

<section class="slide" data-screen-label="05 Training">
  <div class="section-label"><div class="num">05</div><div><div class="kicker">Training Route</div><div class="line"></div></div></div>
  <h2>训练先稳定“病害前景”，再迁移到 38 类细分</h2>
  <p class="lead">当前更稳的路线是先做二分类，校准前景阈值和边界，再把 backbone 迁移到细粒度类别体系。</p>
  <div class="train-map">
    <div class="stages">
      <div class="stage"><span>01</span><div><h3>二分类</h3><p>背景 / 病害，先解决能否识别前景。</p></div></div>
      <div class="stage"><span>02</span><div><h3>阈值校准</h3><p>当前经验阈值约在 0.44 附近更优。</p></div></div>
      <div class="stage"><span>03</span><div><h3>迁移骨干</h3><p>复用前景边界和几何表征。</p></div></div>
      <div class="stage"><span>04</span><div><h3>38 类细分</h3><p>分阶段扩展，避免冷启动塌缩。</p></div></div>
    </div>
    <div class="architecture">
      <div class="tag">MODEL PIPELINE</div>
      <div class="arch-row">
        <div class="arch-box"><h3>输入点云</h3><p>points / feats / normals</p></div>
        <div class="arch-arrow">→</div>
        <div class="arch-box"><h3>Swin3D</h3><p>当前稳定精度基线</p></div>
        <div class="arch-arrow">→</div>
        <div class="arch-box"><h3>mHC</h3><p>通道混合增强</p></div>
        <div class="arch-arrow">→</div>
        <div class="arch-box"><h3>Seg Head</h3><p>二分类 / 多分类输出</p></div>
      </div>
      <div class="loss-band">
        <div class="loss-item"><b>Focal</b><p>抑制背景主导</p></div>
        <div class="loss-item"><b>Dice</b><p>稳定小目标</p></div>
        <div class="loss-item"><b>Edge</b><p>强化裂缝边界</p></div>
        <div class="loss-item"><b>Muon + AdamW</b><p>混合优化，可回退</p></div>
      </div>
    </div>
  </div>
  <div class="folio">05</div>
</section>

<section class="slide" data-screen-label="06 Evidence">
  <div class="section-label"><div class="num">06</div><div><div class="kicker">Current Evidence</div><div class="line"></div></div></div>
  <h2>当前结果说明路线可行，但不能被说成最终泛化结论</h2>
  <div class="evidence-layout">
    <div class="metric-wall">
      <div class="metric-large"><div class="metric-value">0.7319</div><div class="metric-caption">Swin3D + mHC 校准后 global mIoU</div></div>
      <div class="metric-large"><div class="metric-value">0.7052</div><div class="metric-caption">disease IoU</div></div>
      <div class="metric-small"><div class="metric-value">0.9039</div><div class="metric-caption">precision</div></div>
      <div class="metric-small"><div class="metric-value">0.7624</div><div class="metric-caption">recall</div></div>
      <div class="metric-small"><div class="metric-value">0.5671</div><div class="metric-caption">PointMamba + mHC 短跑 mIoU</div></div>
      <div class="metric-small"><div class="metric-value">5000</div><div class="metric-caption">下一步目标合成场景规模</div></div>
    </div>
    <aside class="evidence-note">
      <h3>可以说：二分类路线已经从“能不能学”进入“如何提稳”。</h3>
      <p>Swin3D 当前更稳；PointMamba 是轻量候选；mHC 在当前主线中保留为通道混合模块。</p>
      <div class="caveat">边界：这些结果来自生成的小型验证 / 测试集，部分快速诊断 batch 很少，存在过拟合可能。</div>
    </aside>
  </div>
  <div class="folio">06</div>
</section>

<section class="slide" data-screen-label="07 GAN End2End">
  <div class="section-label"><div class="num">07</div><div><div class="kicker">Domain Adaptation Branch</div><div class="line"></div></div></div>
  <h2>GAN 和 end2end 是走向真实域的接口，不是当前主结果</h2>
  <p class="lead">代码中的 gan_enhanced / end2end 需要讲，但定位要准确：它们服务于合成到真实风格的域适配，主结果仍应先由稳定基线支撑。</p>
  <div class="gan-map">
    <div class="main-branch">
      <div class="branch-title">MAINLINE FIRST</div>
      <div class="branch-flow">
        <div class="branch-box"><h3>合成点云</h3><p>可控几何和逐点标签</p></div>
        <div class="branch-arrow">→</div>
        <div class="branch-box"><h3>稳定训练</h3><p>Swin3D + mHC 二分类基线</p></div>
        <div class="branch-arrow">→</div>
        <div class="branch-box"><h3>正式验证</h3><p>统一 global mIoU 口径</p></div>
      </div>
    </div>
    <div class="future-branch">
      <div class="branch-title">FUTURE INTERFACE</div>
      <div class="future-list">
        <p>StyleTransferGen 对坐标和法向量做风格残差调整。</p>
        <p>WGAN 判别器判断点云风格是否更接近目标域。</p>
        <p>end2end 把风格调整与分割训练放入同一循环。</p>
        <p>主线稳定后，再用它验证合成到真实的域适配能力。</p>
      </div>
    </div>
  </div>
  <div class="folio">07</div>
</section>

<section class="slide" data-screen-label="08 Risks">
  <div class="section-label"><div class="num">08</div><div><div class="kicker">Research Risks</div><div class="line"></div></div></div>
  <h2>现在最该补的不是继续堆模块，而是扩大验证和统一口径</h2>
  <p class="lead">项目已经跑通主线，但要往论文或比赛成果推进，必须先解决可信度问题。</p>
  <div class="risk-matrix">
    <div class="risk-column"><div class="index">01</div><h3>数据规模</h3><p>小型合成验证集容易让模型在少数 batch 上过拟合。</p><div class="fix">扩大到约 5000 个合成场景，固定生成元数据。</div></div>
    <div class="risk-column"><div class="index">02</div><h3>评估口径</h3><p>batch mean mIoU 与全验证集 global mIoU 需要统一记录。</p><div class="fix">每轮保存 global mIoU、disease IoU、precision、recall。</div></div>
    <div class="risk-column"><div class="index">03</div><h3>38 类迁移</h3><p>从二分类直接跳到 38 类过陡，类别长尾会放大不稳定。</p><div class="fix">二分类 → 高频子类 → 38 类，分阶段迁移。</div></div>
    <div class="risk-column"><div class="index">04</div><h3>真实域差异</h3><p>真实点云噪声、密度和采集角度仍未形成正式闭环。</p><div class="fix">主线稳定后，再接 GAN / end2end 和真实点云验证。</div></div>
  </div>
  <div class="folio">08</div>
</section>

<section class="slide" data-screen-label="09 Next Discussion">
  <div class="section-label"><div class="num">09</div><div><div class="kicker">Advisor Discussion</div><div class="line"></div></div></div>
  <h2>下一阶段围绕“扩数据、稳基线、分阶段多类”推进</h2>
  <div class="next-layout">
    <div class="roadmap">
      <div class="roadmap-row"><span>01</span><h3>扩大合成数据</h3><p>目标约 5000 个场景，优先保证可恢复生成、分批执行和统一元数据。</p></div>
      <div class="roadmap-row"><span>02</span><h3>重做正式验证</h3><p>固定 Swin3D + mHC 二分类基线，统一 global mIoU、disease IoU、precision、recall。</p></div>
      <div class="roadmap-row"><span>03</span><h3>分阶段回到 38 类</h3><p>二分类 → 高频子类 / 4-8 类 → 38 类，避免冷启动直接塌缩。</p></div>
      <div class="roadmap-row"><span>04</span><h3>再接真实域</h3><p>主线稳定后，再接 GAN / end2end 和真实点云验证。</p></div>
    </div>
    <aside class="decision">
      <h3>希望导师判断的核心问题</h3>
      <p>这个项目下一步优先按论文路线推进，还是先按比赛 / 演示路线包装？两条路线需要不同的验证密度和交付重点。</p>
      <div class="choice">
        <div><b>A</b>论文路线：强调方法闭环、消融、数据规模和泛化验证。</div>
        <div><b>B</b>展示路线：强调可视化、稳定 demo、生成样例和端到端流程。</div>
      </div>
    </aside>
  </div>
  <div class="folio">09</div>
</section>
</main>
<script>
const params = new URLSearchParams(location.search);
const target = Number(params.get("slide"));
if (Number.isFinite(target)) {
  document.querySelectorAll(".slide").forEach((el, idx) => {
    el.style.display = idx === target ? "block" : "none";
  });
}
</script>
</body>
</html>`;

await fs.mkdir(outDir, { recursive: true });
await fs.mkdir(pngDir, { recursive: true });
await fs.writeFile(htmlPath, html, "utf8");

const fileUrl = pathToFileURL(htmlPath).href;
for (let i = 0; i < notes.length; i += 1) {
  const screenshotPath = path.join(pngDir, `slide-${String(i + 1).padStart(2, "0")}.png`);
  const result = spawnSync(chromePath, [
    "--headless=new",
    "--disable-gpu",
    "--hide-scrollbars",
    "--allow-file-access-from-files",
    "--force-device-scale-factor=1",
    "--window-size=1920,1080",
    `--screenshot=${screenshotPath}`,
    `${fileUrl}?slide=${i}`
  ], { encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(`Chrome screenshot failed for slide ${i + 1}:\n${result.stderr || result.stdout}`);
  }
}

const pptx = new pptxgen();
pptx.author = "Codex";
pptx.subject = "RoadMC 导师汇报";
pptx.title = "RoadMC 导师汇报设计版";
pptx.company = "RoadMC";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Noto Serif SC",
  bodyFontFace: "Noto Sans SC",
  lang: "zh-CN"
};
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";

for (let i = 0; i < notes.length; i += 1) {
  const slide = pptx.addSlide();
  slide.background = { color: "F5EADC" };
  slide.addImage({
    path: path.join(pngDir, `slide-${String(i + 1).padStart(2, "0")}.png`),
    x: 0,
    y: 0,
    w: 13.333333,
    h: 7.5
  });
  if (typeof slide.addNotes === "function") slide.addNotes(notes[i]);
}

await pptx.writeFile({ fileName: pptxPath });
console.log(JSON.stringify({ htmlPath, pptxPath, pngDir, slideCount: notes.length }, null, 2));
