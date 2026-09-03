import fs from "node:fs/promises";
import path from "node:path";
import {
  FileBlob,
  PresentationFile,
} from "@oai/artifact-tool";

const workspace =
  "C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\.codex\\tmp\\paper_detail_deck";
const starterPath = path.join(workspace, "template-starter.pptx");
const outputPath =
  "C:\\Users\\siomi\\OneDrive\\デスクトップ\\第4回　カメラ班ミーティング_2026-07-31_論文説明拡充.pptx";
const renderDir = path.join(workspace, "final-render");
const layoutDir = path.join(workspace, "final-layout");
const sourceDetailTitle = "元論文はドッキングリング内周を3段階で検出";
const sourceDetailBody = [
  "【目的】",
  "宇宙機画像から、ドッキングリング内周の楕円パラメータを高速に求める",
  "",
  "【処理】",
  "① Cannyエッジから弧を抽出し、勾配方向と凸性で4象限に分類",
  "② 位置・接線・中心の幾何制約を満たす三弧を同一楕円として統合",
  "③ 最小二乗で楕円を推定し、弧点との適合度で候補を検証",
  "",
  "【論文中の評価】",
  "実画像31枚で検出性能を比較し、平均処理時間27.38 msを報告",
].join("\n");

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function findRecord(records, slideNumber, kind, name) {
  const record = records.find(
    (candidate) =>
      candidate.slide === slideNumber
      && candidate.kind === kind
      && (name === undefined || candidate.name === name),
  );
  if (!record) {
    throw new Error(
      `編集対象が見つかりません: slide=${slideNumber}, kind=${kind}, name=${name}`,
    );
  }
  return record;
}

function setSlideText(presentation, records, slideNumber, title, body) {
  const titleShape = presentation.resolve(
    findRecord(records, slideNumber, "textbox", "テキスト ボックス 6").id,
  );
  const bodyShape = presentation.resolve(
    findRecord(records, slideNumber, "textbox", "テキスト ボックス 1").id,
  );
  titleShape.text.replace(sourceDetailTitle, title);
  bodyShape.text = body;
  bodyShape.text.style = {
    fontSize: 26.67,
    typeface: "Yu Gothic",
    color: "tx1",
    alignment: "left",
    autoFit: "resizeShapeToFitText",
    wrap: "none",
  };

  // 主要式を本文と同じ書式のまま太字にし、発表時に追いやすくする。
  for (const formula of [
    "(dₓ, dᵧ)",
    "D(eᵢ) = {",
    "Ax² + By² + Cxy + Dx + Ey + F = 0",
    "s > Tₛ",
  ]) {
    if (body.includes(formula)) {
      bodyShape.text.get(formula).bold = true;
    }
  }
}

function setNotes(presentation, records, slideNumber, lines) {
  presentation.resolve(
    findRecord(records, slideNumber, "notes").id,
  ).setText(lines.join("\n"));
}

async function main() {
  const presentation = await PresentationFile.importPptx(
    await FileBlob.load(starterPath),
  );
  const before = await presentation.inspect({
    kind: "textbox,notes",
    maxChars: 50000,
  });
  const records = before.ndjson
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));

  setSlideText(
    presentation,
    records,
    4,
    "① 勾配方向と凸性で楕円らしい円弧だけを残す",
    [
      "まずCanny法でエッジ点を抽出する。",
      "各点 eᵢ のSobel勾配を (dₓ, dᵧ) とし、符号で4方向に分類する。",
      "",
      "D(eᵢ) = {",
      "　Ⅰ　dₓ > 0,  dᵧ > 0",
      "　Ⅱ　dₓ < 0,  dᵧ > 0",
      "　Ⅲ　dₓ < 0,  dᵧ < 0",
      "　Ⅳ　dₓ > 0,  dᵧ < 0",
      "",
      "同じ分類の8近傍点を連結して円弧にし、短すぎるものを除外する。",
      "さらに弦に対する膨らみ方から凸・凹を判定し、",
      "楕円境界の上下左右と矛盾する円弧を除く。",
    ].join("\n"),
  );

  setSlideText(
    presentation,
    records,
    5,
    "② 幾何制約を満たす3円弧を1つの楕円にまとめる",
    [
      "遮蔽やノイズで輪郭が切れても、完全な4円弧は要求しない。",
      "隣接する3領域の円弧を、次の条件で絞り込む。",
      "",
      "・隣接する勾配領域に属する",
      "・端点の位置が楕円の並びと矛盾しない",
      "・端点の接線に対し、相手円弧が楕円の内側にある",
      "・各円弧から推定した中心が互いに近い",
      "",
      "条件を通った3円弧の全点を、一般二次曲線",
      "",
      "Ax² + By² + Cxy + Dx + Ey + F = 0",
      "",
      "に最小二乗で当てはめ、楕円の中心・回転角・長短軸を求める。",
    ].join("\n"),
  );

  setSlideText(
    presentation,
    records,
    6,
    "③ 弧点との一致度が高い楕円候補だけを残す",
    [
      "最小二乗だけでは、背景輪郭にも楕円が当てはまる可能性がある。",
      "そこで候補楕円と、統合に使った3円弧の一致を確認する。",
      "",
      "・各弧点から楕円中心へ向かう直線と、楕円境界の交点を求める",
      "・弧点と交点が近ければ「適合点」と数える",
      "・3円弧全体に対する適合点の割合を一致度 s とする",
      "",
      "s > Tₛ",
      "",
      "を満たす候補だけを残し、ほぼ同じ楕円なら一致度が高い方を採用する。",
      "全周を総当たりせず、円弧分類と幾何制約で候補を絞るのが要点。",
    ].join("\n"),
  );

  setNotes(
    presentation,
    records,
    3,
    [
      "想定35秒。",
      "元論文の狙いは、画像上で楕円になったドッキングリング内周を高速に検出すること。",
      "処理は円弧抽出、三円弧統合、楕円候補の検証の3段階。",
      "次の3枚で、式を含めて順に説明する。",
      "",
      "[Sources]",
      "- https://doi.org/10.3390/s19235243",
      "- https://pmc.ncbi.nlm.nih.gov/articles/PMC6928708/",
    ],
  );

  setNotes(
    presentation,
    records,
    4,
    [
      "想定45秒。",
      "まずCannyでエッジ点を得て、Sobel勾配の符号から4方向に分類する。",
      "同じ分類の近傍点をつないで円弧を作り、短い円弧を除く。",
      "端点間の弦に対して円弧がどちら側へ膨らむかを調べ、楕円の上下左右と矛盾する円弧も除外する。",
      "",
      "[Sources]",
      "- https://doi.org/10.3390/s19235243",
      "- https://pmc.ncbi.nlm.nih.gov/articles/PMC6928708/",
    ],
  );

  setNotes(
    presentation,
    records,
    5,
    [
      "想定50秒。",
      "遮蔽を想定して完全な4方向の円弧は要求せず、隣接する3方向を組み合わせる。",
      "端点位置、接線に対する相手円弧の位置、各円弧から推定した中心の近さで候補を絞る。",
      "条件を通った3円弧の全点を一般二次曲線へ最小二乗で当てはめ、楕円パラメータを求める。",
      "",
      "[Sources]",
      "- https://doi.org/10.3390/s19235243",
      "- https://pmc.ncbi.nlm.nih.gov/articles/PMC6928708/",
    ],
  );

  setNotes(
    presentation,
    records,
    6,
    [
      "想定35秒。",
      "当てはめだけでは背景輪郭の誤検出が残るため、元の弧点と候補楕円の一致度を確認する。",
      "中心から見た楕円境界と弧点が近いものを適合点とし、その割合が閾値を超えた候補だけを残す。",
      "赤線で削除指定された距離式とスコア定義式はスライドでは省略し、考え方だけを示している。",
      "",
      "[Sources]",
      "- https://doi.org/10.3390/s19235243",
      "- https://pmc.ncbi.nlm.nih.gov/articles/PMC6928708/",
    ],
  );

  setNotes(
    presentation,
    records,
    7,
    [
      "想定45秒。",
      "ここまでの3段階を、再現実装の中間画像で確認する。",
      "Cannyエッジから円弧を分類し、幾何制約を満たす三円弧を統合して楕円を当てる。",
      "代表例では推定とCG正解のIoUは0.995。",
      "",
      "[Sources]",
      "- https://doi.org/10.3390/s19235243",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\paf_ring_detection\\methods\\zhang2019.py",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\.codex\\tmp\\camera_meeting_4_20260731\\generated-assets\\zhang_visual_metrics.json",
    ],
  );

  setNotes(
    presentation,
    records,
    8,
    [
      "想定45秒。",
      "著者コードの移植ではなく、論文の処理構成をPythonで再現したもの。",
      "論文処理で候補を作った後、PAF向けの最終選択には2規則だけを使う。",
      "内包関係があれば最も内側、内包関係がなければ全周支持率が最大の候補を選ぶ。",
      "",
      "[Sources]",
      "- https://doi.org/10.3390/s19235243",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\paf_ring_detection\\methods\\zhang2019.py",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\paf_ring_detection\\methods\\zhang2019_paf.py",
    ],
  );

  setNotes(
    presentation,
    records,
    9,
    [
      "想定35秒。",
      "評価は撮像診断1,456枚に一本化した。",
      "基礎画像112枚へCleanと3種類の撮像劣化を4強度で適用した。",
      "3方式を同じ画像、同じ成功基準の楕円IoU 0.80以上で比較した。",
      "",
      "[Sources]",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\config\\experiment.json",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\datasets\\diagnostic_evaluation\\manifest.json",
    ],
  );

  setNotes(
    presentation,
    records,
    10,
    [
      "想定45秒。",
      "撮像診断1,456枚では、Zhang 2019型（再現実装）は693枚成功で47.6%。",
      "Canny統制の46.8%とはほぼ同等。",
      "CNN方式は74.5%で、Zhang型より26.9ポイント高い。",
      "",
      "[Sources]",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\summary.json",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\diagnostic\\canny_contour_ransac.csv",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\diagnostic\\zhang2019.csv",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\diagnostic\\cnn_ransac.csv",
    ],
  );

  setNotes(
    presentation,
    records,
    11,
    [
      "想定45秒。",
      "白飛びは強度100%でも83.9%。",
      "黒矩形は50%で3.6%、75%以上で0%。",
      "黒つぶれは75%で20.5%、100%で8.9%。",
      "必要な円弧そのものが消える条件では、三円弧統合が成立しにくい。",
      "",
      "[Sources]",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\diagnostic\\zhang2019.csv",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\diagnostic_by_severity.png",
    ],
  );

  setNotes(
    presentation,
    records,
    12,
    [
      "想定25秒。",
      "Zhang 2019型（再現実装）は撮像診断で47.6%。",
      "Canny統制と同等で、CNNより低かった。",
      "遮蔽や黒つぶれで円弧自体が不足することが主な失敗要因。",
      "未知条件への一般化は次回以降に扱う。",
      "",
      "[Sources]",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\.codex\\PROJECT_CONTEXT.md",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\summary.json",
    ],
  );

  await fs.mkdir(renderDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });
  for (const [index, slide] of presentation.slides.items.entries()) {
    const number = String(index + 1).padStart(2, "0");
    await writeBlob(
      path.join(renderDir, `slide-${number}.png`),
      await presentation.export({ slide, format: "png", scale: 1 }),
    );
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(
      path.join(layoutDir, `slide-${number}.layout.json`),
      await layout.text(),
      "utf8",
    );
  }

  const montage = await presentation.export({
    format: "webp",
    montage: true,
    scale: 1,
  });
  await writeBlob(path.join(workspace, "final-montage.webp"), montage);

  const inspect = await presentation.inspect({
    kind: "slide,textbox,shape,image,table,chart,notes,layout",
    maxChars: 50000,
  });
  await fs.writeFile(
    path.join(workspace, "final-inspect.ndjson"),
    inspect.ndjson,
    "utf8",
  );

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(outputPath);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
