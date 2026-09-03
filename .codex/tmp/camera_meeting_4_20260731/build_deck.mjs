import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(scriptDir, "..", "..", "..");
const starterPath = path.join(scriptDir, "template-starter.pptx");
const finalPath =
  "C:\\Users\\siomi\\OneDrive\\デスクトップ\\第4回　カメラ班ミーティング_2026-07-31_評価一本化.pptx";
const renderDir = path.join(scriptDir, "final-renders");
const layoutDir = path.join(scriptDir, "final-layout");
const assetDir = path.join(scriptDir, "generated-assets");
const starterInspectPath = path.join(scriptDir, "build-alias-inspect.ndjson");
const semanticSpecs = {
  S6_TITLE: {
    kind: "textbox",
    slide: 6,
    name: "テキスト ボックス 6",
    text: "Blenderを用いた初期データセットの作成",
    bbox: [26.18, 61.09, 819.96, 61.39],
  },
  S6_IMAGE_OOD: {
    kind: "image",
    slide: 6,
    name: "図 16",
    bbox: [155.99, 158.7, 206.19, 206.19],
  },
  S6_IMAGE_BLACK_RECT: {
    kind: "image",
    slide: 6,
    name: "図 22",
    bbox: [607.14, 162.18, 145.41, 145.41],
  },
  S6_IMAGE_WHITEOUT: {
    kind: "image",
    slide: 6,
    name: "図 20",
    bbox: [787.13, 162.18, 145.41, 145.41],
  },
  S6_IMAGE_BLACK_CRUSH: {
    kind: "image",
    slide: 6,
    name: "図 18",
    bbox: [967.13, 162.11, 145.41, 145.41],
  },
  S6_IMAGE_OLD_TRUTH: {
    kind: "image",
    slide: 6,
    name: "図 46",
    bbox: [175.76, 526.72, 194.04, 122.53],
  },
  S6_LABEL_OOD: {
    kind: "textbox",
    slide: 6,
    name: "テキスト ボックス 27",
    text: "基礎データ",
    bbox: [188.81, 125.65, 140.56, 38.78],
  },
  S6_LABEL_BLACK_RECT: {
    kind: "textbox",
    slide: 6,
    name: "テキスト ボックス 23",
    text: "遮蔽",
    bbox: [645.91, 123.33, 67.86, 38.78],
  },
  S6_LABEL_WHITEOUT: {
    kind: "textbox",
    slide: 6,
    name: "テキスト ボックス 29",
    text: "白飛び",
    bbox: [813.79, 122.48, 92.09, 38.78],
  },
  S6_LABEL_BLACK_CRUSH: {
    kind: "textbox",
    slide: 6,
    name: "テキスト ボックス 31",
    text: "黒つぶれ",
    bbox: [981.67, 122.48, 116.33, 38.78],
  },
  S6_OOD_CAPTION: {
    kind: "textbox",
    slide: 6,
    name: "テキスト ボックス 25",
    text: "102カメラ条件×7照明条件=714枚",
    bbox: [123.42, 372.62, 271.32, 29.08],
  },
  S6_COUNT_OOD: {
    kind: "textbox",
    slide: 6,
    name: "四角形: 角を丸くする 32",
    text: "学習 497枚",
    bbox: [420.35, 180.23, 133.61, 48.07],
  },
  S6_COUNT_DIAGNOSTIC: {
    kind: "textbox",
    slide: 6,
    name: "四角形: 角を丸くする 34",
    text: "学習時検証 105枚",
    bbox: [420.35, 242.48, 133.61, 48.48],
  },
  S6_SUCCESS: {
    kind: "textbox",
    slide: 6,
    name: "四角形: 角を丸くする 36",
    text: "テスト112枚",
    bbox: [420.35, 305.15, 133.61, 48.48],
  },
  S6_OLD_SPLIT_DETAIL: {
    kind: "textbox",
    slide: 6,
    name: "テキスト ボックス 40",
    text: "・学習：497枚×(元画像+3劣化) = 1988枚\n・検証：105枚×(元画像+3劣化×2強度) = 735枚\n・テスト：112枚×(元画像+3劣化×4強度) = 1456枚",
    bbox: [662.4, 319.3, 403.27, 67.86],
  },
  S6_BODY_OOD: {
    kind: "textbox",
    slide: 6,
    name: "テキスト ボックス 1",
    text: "【基礎データ】\n・102カメラ条件 × 7照明条件 = 714枚\n・正解データ：PAF内周輪郭",
    bbox: [85.77, 402.71, 497.85, 106.63],
  },
  S6_BODY_DIAGNOSTIC: {
    kind: "textbox",
    slide: 6,
    name: "テキスト ボックス 2",
    text: "【データ分割と劣化効果の付与】\n・学習・テストで異なるカメラ条件を使用\n・OpenCVで遮蔽／白飛び／黒つぶれを付与",
    bbox: [662.4, 401.7, 552.04, 106.63],
  },
  S6_OLD_TRUTH_TEXT: {
    kind: "textbox",
    slide: 6,
    name: "テキスト ボックス 1",
    text: "サイズ：256×256\n正解楕円幅：3 px",
    bbox: [379.03, 550.83, 243.89, 74.32],
  },
  S7_TITLE: {
    kind: "textbox",
    slide: 7,
    name: "テキスト ボックス 6",
    text: "比較方法",
    bbox: [26.18, 82.27, 191.72, 61.39],
  },
  S7_BODY1: {
    kind: "textbox",
    slide: 7,
    name: "テキスト ボックス 1",
    text: "【評価条件】\n・Clean、25、50、75、100%\n・各条件112枚",
    bbox: [12.71, 214.25, 454.09, 126.02],
  },
  S7_BODY2: {
    kind: "textbox",
    slide: 7,
    name: "テキスト ボックス 2",
    text: "【成功条件】\n・推定楕円と正解楕円のIoU ≥ 0.80\n・同一元画像で3方式を比較",
    bbox: [12.71, 380.01, 530.67, 126.02],
  },
  S7_IMAGE: {
    kind: "image",
    slide: 7,
    name: "図 15",
    bbox: [579.51, 170.42, 674.31, 379.17],
  },
};

let templateRecords = [];
let runtimeRecords = [];

function parseNdjson(text) {
  return text
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function sameBbox(first, second) {
  if (!Array.isArray(first) || !Array.isArray(second) || first.length !== second.length) {
    return false;
  }
  return first.every((value, index) => Math.abs(value - second[index]) < 0.2);
}

async function initializeRuntimeResolver(presentation) {
  templateRecords = parseNdjson(
    await fs.readFile(starterInspectPath, "utf8"),
  );
  const snapshot = await presentation.inspect({
    kind: "slide,textbox,shape,image,notes",
    maxChars: 200000,
  });
  runtimeRecords = parseNdjson(snapshot.ndjson ?? "");
}

function runtimeId(aliasId) {
  const template =
    semanticSpecs[aliasId] ??
    templateRecords.find((record) => record.id === aliasId);
  if (!template) {
    throw new Error(`テンプレート検査に存在しない要素IDです: ${aliasId}`);
  }

  const exact = runtimeRecords.find(
    (record) =>
      record.kind === template.kind &&
      record.slide === template.slide &&
      record.name === template.name &&
      record.text === template.text &&
      sameBbox(record.bbox, template.bbox),
  );
  if (exact) return exact.id;

  const candidates = runtimeRecords.filter(
    (record) =>
      record.kind === template.kind &&
      record.slide === template.slide &&
      record.name === template.name,
  );
  if (candidates.length === 1) return candidates[0].id;

  const byText = candidates.find((record) => record.text === template.text);
  if (byText) return byText.id;

  const byBox = candidates.find((record) => sameBbox(record.bbox, template.bbox));
  if (byBox) return byBox.id;

  throw new Error(
    `実行時要素を照合できませんでした: ${aliasId} slide=${template.slide} name=${template.name}`,
  );
}

function resolveElement(presentation, aliasId) {
  return presentation.resolve(runtimeId(aliasId));
}

async function readImageBytes(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function rewrite(presentation, id, newText, position) {
  const shape = resolveElement(presentation, id);
  // 元のテキストスタイルを継承したまま、内容だけを入れ替える。
  shape.text = newText;
  if (position) shape.position = position;
}

async function replaceImage(presentation, id, imagePath, alt) {
  const image = resolveElement(presentation, id);
  const oldFrame = image.frame;
  const oldGeometry = image.geometry;
  const oldBorderRadius = image.borderRadius;
  const oldRotation = image.rotation;
  const oldFlipHorizontal = image.flipHorizontal;
  const oldFlipVertical = image.flipVertical;
  const oldLockAspectRatio = image.lockAspectRatio;

  image.replace({
    blob: await readImageBytes(imagePath),
    contentType: "image/png",
    alt,
    fit: "contain",
  });
  image.frame = oldFrame;
  image.crop = { left: 0, top: 0, right: 0, bottom: 0 };
  image.geometry = oldGeometry;
  image.borderRadius = oldBorderRadius;
  image.rotation = oldRotation;
  image.flipHorizontal = oldFlipHorizontal;
  image.flipVertical = oldFlipVertical;
  image.lockAspectRatio = oldLockAspectRatio;
  image.fit = "contain";
}

function setNotes(presentation, slideId, lines) {
  const slide = resolveElement(presentation, slideId);
  slide.speakerNotes.textFrame.setText(lines.join("\n"));
  slide.speakerNotes.setVisible(true);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(renderDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });

  const presentation = await PresentationFile.importPptx(
    await FileBlob.load(starterPath),
  );
  await initializeRuntimeResolver(presentation);

  // 1. 表紙
  rewrite(presentation, "sh/6twjuhg7", "第4回　カメラ班ミーティング");
  rewrite(presentation, "sh/7u50nmxs", "2026年7月31日\nB4　伊藤弘道");
  setNotes(presentation, "sl/xoevpu", [
    "想定20秒。",
    "今回はZhang 2019型（再現実装）の導入、仕組み、評価結果を報告する。",
    "",
    "[Sources]",
    "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\第3回　カメラ班ミーティング.pptx（デザイン参照）",
  ]);

  // 2. 再現元論文
  rewrite(
    presentation,
    "sh/n2dwf6t4",
    "Real-Time Docking Ring Detection Based on the\nGeometrical Shape for an On-Orbit Spacecraft\n\nZhang, Pan, Ma／Sensors 2019\nDOI: 10.3390/s19235243",
    { left: 50, top: 238.83, width: 1180, height: 326.4 },
  );
  setNotes(presentation, "sl/vf4gdb", [
    "想定25秒。",
    "今回再現したのは、Zhang、Pan、MaがSensorsに発表した2019年の論文。",
    "宇宙機のドッキングリングを、画像上の楕円としてリアルタイム検出する手法を提案している。",
    "",
    "[Sources]",
    "- https://doi.org/10.3390/s19235243",
    "- https://pmc.ncbi.nlm.nih.gov/articles/PMC6928708/",
  ]);

  // 3. 元論文の要約
  rewrite(
    presentation,
    "sh/l8napofa",
    "元論文はドッキングリング内周を3段階で検出",
    { left: 26.18, top: 61.09, width: 1180, height: 61.39 },
  );
  rewrite(
    presentation,
    "sh/upsr2pgf",
    "【目的】\n宇宙機画像から、ドッキングリング内周の楕円パラメータを高速に求める\n\n【処理】\n① Cannyエッジから弧を抽出し、勾配方向と凸性で4象限に分類\n② 位置・接線・中心の幾何制約を満たす三弧を同一楕円として統合\n③ 最小二乗で楕円を推定し、弧点との適合度で候補を検証\n\n【論文中の評価】\n実画像31枚で検出性能を比較し、平均処理時間27.38 msを報告",
    { left: 70, top: 172, width: 1110, height: 420 },
  );
  resolveElement(presentation, "im/254nqh43").delete();
  setNotes(presentation, "sl/h5ne2e", [
    "想定65秒。",
    "論文の目的は、宇宙機画像からドッキングリング内周の楕円パラメータを高速に求めること。",
    "処理は、弧の抽出と分類、三弧の統合、楕円推定と検証の3段階。",
    "分断された輪郭を連結成分ごとに扱うのではなく、幾何制約を満たす別々の弧を同じ楕円へまとめる点が特徴。",
    "論文では実画像31枚で既存手法と比較し、平均処理時間27.38ミリ秒と報告している。",
    "",
    "[Sources]",
    "- https://doi.org/10.3390/s19235243",
    "- https://pmc.ncbi.nlm.nih.gov/articles/PMC6928708/",
  ]);

  // 4. 再現実装の中間処理
  rewrite(
    presentation,
    "sh/n2xoz6hk",
    "三弧統合で分断エッジから内周楕円を復元",
    { left: 26.18, top: 82.27, width: 1180, height: 61.39 },
  );
  await replaceImage(
    presentation,
    "im/h0nmpwrq",
    path.join(assetDir, "zhang_process_strip.png"),
    "Zhang 2019型（再現実装）の入力、Canny、弧分類、三弧統合、最終選択",
  );
  resolveElement(presentation, "sh/kjml47ah").delete();
  resolveElement(presentation, "im/udc3uhsz").delete();
  resolveElement(presentation, "sh/yhk32xsb").delete();
  rewrite(
    presentation,
    "sh/wf2l0nal",
    "再現実装の中間処理（代表例）",
  );
  rewrite(
    presentation,
    "sh/fqt4jids",
    "・Canny後、勾配象限と凸性が整合する弧だけを残す\n・位置・接線制約と中心整合により、同一楕円らしい三弧を統合",
  );
  rewrite(
    presentation,
    "sh/b6ho3qhw",
    "・統合弧へ最小二乗楕円を当て、中心光線距離による適合点率で検証",
  );
  setNotes(presentation, "sl/mb7o1w", [
    "想定85秒。",
    "代表例は実際の再現実装から出力した中間処理。",
    "CannyエッジにはPAF外周や放射状構造も含まれる。",
    "勾配象限と凸性で弧を分類し、隣接象限の位置・接線条件を満たす組合せだけを残す。",
    "三弧から別々に推定した中心が近い場合に統合し、最小二乗楕円を当てる。",
    "代表例では推定とCG正解のIoUは0.995。",
    "",
    "[Sources]",
    "- https://doi.org/10.3390/s19235243",
    `- ${path.join(projectDir, "paf_ring_detection", "methods", "zhang2019.py")}`,
    `- ${path.join(assetDir, "zhang_visual_metrics.json")}`,
  ]);

  // 5. 再現実装の範囲
  rewrite(
    presentation,
    "sh/sf29gned",
    "Zhang 2019型（再現実装）は候補生成＋PAF内周選択",
    { left: 26.18, top: 82.27, width: 1180, height: 61.39 },
  );
  rewrite(
    presentation,
    "sh/lorax47u",
    "論文の処理で候補を作り、PAF固有条件で1つに絞る",
    { left: 230, top: 143.66, width: 960, height: 48.47 },
  );
  resolveElement(presentation, "im/0ve9sj2d").delete();
  await replaceImage(
    presentation,
    "im/lwna1o3y",
    path.join(assetDir, "zhang_paf_validation_wide.png"),
    "楕円全周のエッジ支持点。緑が支持あり、赤が支持なし",
  );
  resolveElement(presentation, "im/lwna1o3y").frame = {
    left: 90,
    top: 192.13,
    width: 1090,
    height: 228,
  };
  rewrite(
    presentation,
    "sh/zm98vupo",
    "① 候補生成（論文）：Canny → 弧分類 → 三弧統合 → 楕円当てはめ\n② 最終選択（追加）：全周支持・明暗極性・同心候補から内周を採用",
  );
  rewrite(
    presentation,
    "sh/xkrqt47y",
    "著者コードの移植ではなく、論文の処理構成を参考にしたPython再現実装。",
  );
  setNotes(presentation, "sl/4qmzdt", [
    "想定60秒。",
    "成果物では必ずZhang 2019型（再現実装）と表記する。",
    "著者コードの移植ではなく、論文の処理構成をPythonで再現したもの。",
    "まず論文と同様に、弧を分類・統合して候補楕円を作る。",
    "次にPAF向けの追加処理として、楕円全周のエッジ支持、内外の明暗、同心候補の内側かどうかを評価し、最終候補を1つ選ぶ。",
    "図の緑は楕円近傍にCannyエッジがある部分、赤はない部分を示す。",
    "",
    "[Sources]",
    "- https://doi.org/10.3390/s19235243",
    `- ${path.join(projectDir, "paf_ring_detection", "methods", "zhang2019.py")}`,
    `- ${path.join(projectDir, "paf_ring_detection", "methods", "zhang2019_paf.py")}`,
  ]);

  // 6. 一本化した撮像診断
  rewrite(
    presentation,
    "S6_TITLE",
    "撮像診断1,456枚で3方式を同条件比較",
    { left: 26.18, top: 61.09, width: 1160, height: 61.39 },
  );
  rewrite(
    presentation,
    "S6_LABEL_OOD",
    "Clean",
  );
  rewrite(
    presentation,
    "S6_LABEL_BLACK_RECT",
    "黒矩形",
  );
  rewrite(presentation, "S6_LABEL_WHITEOUT", "白飛び");
  rewrite(presentation, "S6_LABEL_BLACK_CRUSH", "黒つぶれ");
  rewrite(
    presentation,
    "S6_OOD_CAPTION",
    "学習に未使用のカメラ条件・既知照明",
  );
  rewrite(presentation, "S6_COUNT_OOD", "基礎 112枚");
  rewrite(presentation, "S6_COUNT_DIAGNOSTIC", "診断 1,456枚");
  rewrite(presentation, "S6_SUCCESS", "IoU ≥ 0.80");
  rewrite(
    presentation,
    "S6_BODY_OOD",
    "【基礎画像】\n・学習に未使用の16カメラ条件\n・既知の7照明条件",
  );
  rewrite(
    presentation,
    "S6_BODY_DIAGNOSTIC",
    "【撮像劣化】\n・Clean＋3効果×4強度\n・各効果・各強度112枚",
  );
  await replaceImage(
    presentation,
    "S6_IMAGE_OOD",
    path.join(
      projectDir,
      "output",
      "datasets",
      "diagnostic_evaluation",
      "images",
      "camera_t020_a000_d044.0_o01__light_t000_a000_e03.0__diagnostic__clean_s0000_v00.png",
    ),
    "撮像診断のClean基礎画像",
  );
  await replaceImage(
    presentation,
    "S6_IMAGE_BLACK_RECT",
    path.join(
      projectDir,
      "output",
      "datasets",
      "diagnostic_evaluation",
      "images",
      "camera_t020_a000_d044.0_o01__light_t000_a000_e03.0__diagnostic__black_rectangle_s0750_v03.png",
    ),
    "黒矩形75%の撮像診断画像",
  );
  await replaceImage(
    presentation,
    "S6_IMAGE_WHITEOUT",
    path.join(
      projectDir,
      "output",
      "datasets",
      "diagnostic_evaluation",
      "images",
      "camera_t020_a000_d044.0_o01__light_t000_a000_e03.0__diagnostic__sensor_whiteout_s0750_v07.png",
    ),
    "白飛び75%の撮像診断画像",
  );
  await replaceImage(
    presentation,
    "S6_IMAGE_BLACK_CRUSH",
    path.join(
      projectDir,
      "output",
      "datasets",
      "diagnostic_evaluation",
      "images",
      "camera_t020_a000_d044.0_o01__light_t000_a000_e03.0__diagnostic__sensor_black_crush_s0750_v11.png",
    ),
    "黒つぶれ75%の撮像診断画像",
  );
  for (const id of [
    "S6_IMAGE_OLD_TRUTH",
    "S6_OLD_SPLIT_DETAIL",
    "S6_OLD_TRUTH_TEXT",
  ]) {
    resolveElement(presentation, id).delete();
  }
  setNotes(presentation, "sl/3q7hrt", [
    "想定45秒。",
    "評価は撮像診断1,456枚に一本化した。",
    "基礎画像は学習に使っていない16のカメラ条件と、学習時にも使った7照明条件の組合せで112枚。",
    "112枚それぞれに、Cleanと黒矩形・白飛び・黒つぶれの4強度を適用した。",
    "3方式を同じ画像、同じ成功基準の楕円IoU 0.80以上で比較した。",
    "",
    "[Sources]",
    `- ${path.join(projectDir, "config", "experiment.json")}`,
    `- ${path.join(projectDir, "output", "datasets", "cnn_training", "manifest.json")}`,
    `- ${path.join(projectDir, "output", "datasets", "diagnostic_evaluation", "manifest.json")}`,
  ]);

  // 7. 撮像診断の全体結果
  rewrite(
    presentation,
    "S7_TITLE",
    "撮像診断：Zhang 2019型（再現実装）46.5%、CNN 74.5%",
    { left: 26.18, top: 82.27, width: 1180, height: 61.39 },
  );
  rewrite(
    presentation,
    "S7_BODY1",
    "【1,456枚】\nZhang 2019型（再現実装）\n677枚＝46.5%／Canny 46.8%",
  );
  rewrite(
    presentation,
    "S7_BODY2",
    "CNN 74.5%\n→ 再現実装はCannyと同等\n　CNNより28.0pt低い",
  );
  const chartFrame = resolveElement(presentation, "S7_IMAGE").frame;
  resolveElement(presentation, "S7_IMAGE").delete();
  resolveElement(presentation, "sl/pmhgnr").charts.add("bar", {
    position: chartFrame,
    title: "撮像診断 成功率（IoU ≥ 0.80）",
    titlePlacement: "aboveChart",
    titleTextStyle: { fontSize: 18, fill: "#1f2937", bold: true },
    categories: [""],
    series: [
      {
        name: "Canny＋輪郭別RANSAC",
        values: [46.8],
        fill: "#f39c12",
        valuesFormatCode: '0.0"%"',
      },
      {
        name: "Zhang 2019型（再現実装）",
        values: [46.5],
        fill: "#6b7280",
        valuesFormatCode: '0.0"%"',
      },
      {
        name: "CNN＋RANSAC",
        values: [74.5],
        fill: "#2563eb",
        valuesFormatCode: '0.0"%"',
      },
    ],
    hasLegend: true,
    legend: {
      position: "top",
      overlay: false,
      textStyle: { fontSize: 11, fill: "#374151" },
    },
    barOptions: {
      direction: "column",
      grouping: "clustered",
      gapWidth: 70,
    },
    xAxis: {
      visible: false,
      tickLabelPosition: "none",
      line: { style: "solid", fill: "#ffffff", width: 0 },
      majorGridlines: null,
    },
    yAxis: {
      title: {
        text: "成功率（%）",
        textStyle: { fontSize: 13, fill: "#374151" },
      },
      min: 0,
      max: 100,
      majorUnit: 20,
      numberFormatCode: '0"%"',
      textStyle: { fontSize: 12, fill: "#4b5563" },
      line: { style: "solid", fill: "#9ca3af", width: 1 },
      majorGridlines: { style: "solid", fill: "#e5e7eb", width: 1 },
    },
    dataLabels: {
      showValue: true,
      position: "outEnd",
      textStyle: { fontSize: 13, fill: "#111827", bold: true },
    },
    chartFill: "#ffffff",
    chartLine: { style: "solid", fill: "#ffffff", width: 0 },
    plotAreaFill: "#ffffff",
    plotAreaLine: { style: "solid", fill: "#ffffff", width: 0 },
  });
  setNotes(presentation, "sl/pmhgnr", [
    "想定55秒。",
    "撮像診断1,456枚では、Zhang 2019型（再現実装）は677枚成功で46.5%。",
    "Canny統制の46.8%とはほぼ同等。",
    "CNN方式は74.5%で、Zhang型より28.0ポイント高い。",
    "",
    "[Sources]",
    `- ${path.join(projectDir, "output", "results", "summary.json")}`,
    `- ${path.join(projectDir, "output", "results", "diagnostic", "canny_contour_ransac.csv")}`,
    `- ${path.join(projectDir, "output", "results", "diagnostic", "zhang2019.csv")}`,
    `- ${path.join(projectDir, "output", "results", "diagnostic", "cnn_ransac.csv")}`,
  ]);

  // 8. 劣化別の結果
  rewrite(
    presentation,
    "sh/9obq5wfa",
    "白飛びに強く、遮蔽と黒つぶれに弱い",
    { left: 26.18, top: 82.27, width: 1180, height: 61.39 },
  );
  for (const id of [
    "sh/2187ad4r",
    "sh/bqh87il8",
    "sh/crqpgnmt",
    "sh/ytsrix4z",
    "sh/g3e1sva9",
  ]) {
    resolveElement(presentation, id).delete();
  }
  rewrite(
    presentation,
    "sh/f25kzq9o",
    "・白飛び100%でも83.9%　・黒矩形50%で3.6%、75%以上で0%\n・黒つぶれ75%で14.3%、100%で8.9% → 弧の消失が支配的",
    { left: 26.18, top: 568, width: 1190, height: 86 },
  );
  await replaceImage(
    presentation,
    "im/mdcrqhgz",
    path.join(projectDir, "output", "results", "diagnostic_by_severity.png"),
    "黒矩形、白飛び、黒つぶれの強度別成功率",
  );
  setNotes(presentation, "sl/5dmbqx", [
    "想定65秒。",
    "白飛びは明るくなっても境界勾配が残るため、Zhang型は強度100%でも83.9%。",
    "黒矩形は50%で3.6%、75%以上で0%。隠れた側の弧が不足し、三弧条件を作れない。",
    "黒つぶれも境界が背景へ埋まり、75%で14.3%、100%で8.9%。",
    "幾何拘束は誤候補を減らせる一方、必要な弧そのものが消える条件には弱い。",
    "",
    "[Sources]",
    `- ${path.join(projectDir, "output", "results", "diagnostic", "zhang2019.csv")}`,
    `- ${path.join(projectDir, "output", "results", "diagnostic_by_severity.png")}`,
  ]);

  // 9. まとめと次回以降
  rewrite(
    presentation,
    "sh/ryx43ux4",
    "まとめと次回以降",
    { left: 26.18, top: 82.27, width: 800, height: 61.39 },
  );
  resolveElement(presentation, "sh/u14vy5g3").delete();
  rewrite(
    presentation,
    "sh/zmh47uxg",
    "【今回】\n・Zhang 2019型（再現実装）を導入\n・撮像診断で46.5%（Canny 46.8%、CNN 74.5%）\n・遮蔽／黒つぶれでは弧不足が主な失敗要因",
  );
  rewrite(
    presentation,
    "sh/4v6l8fet",
    "【次回以降】\n・model_generalization_studyで未知条件への一般化を検証\n・両方式の失敗例を条件別に整理",
  );
  setNotes(presentation, "sl/sgjuv0", [
    "想定35秒。",
    "評価は撮像診断に一本化した。",
    "Zhang 2019型（再現実装）は46.5%で、Canny統制と同等、CNNより低かった。",
    "遮蔽や黒つぶれで弧自体が不足すると、三弧統合が成立しない。",
    "model_generalization_studyは今回の本編から外し、次回以降にCNNの未知条件一般化として扱う。",
    "",
    "[Sources]",
    `- ${path.join(projectDir, ".codex", "PROJECT_CONTEXT.md")}`,
    `- ${path.join(projectDir, "output", "results", "summary.json")}`,
  ]);

  const slides = presentation.slides.items;
  for (let index = 0; index < slides.length; index += 1) {
    const slide = slides[index];
    const number = String(index + 1).padStart(2, "0");
    await writeBlob(
      path.join(renderDir, `slide-${number}.png`),
      await presentation.export({ slide, format: "png", scale: 2 }),
    );
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(
      path.join(layoutDir, `slide-${number}.layout.json`),
      await layout.text(),
      "utf8",
    );
  }

  await writeBlob(
    path.join(scriptDir, "final-montage.webp"),
    await presentation.export({ format: "webp", montage: true, scale: 1 }),
  );
  const inspect = await presentation.inspect({
    kind: "slide,textbox,shape,image,notes,layout",
    maxChars: 200000,
  });
  await fs.writeFile(
    path.join(scriptDir, "final-inspect.ndjson"),
    inspect.ndjson ?? "",
    "utf8",
  );

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(finalPath);
  console.log(finalPath);
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
