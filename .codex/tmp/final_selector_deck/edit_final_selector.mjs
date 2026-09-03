import fs from "node:fs/promises";
import path from "node:path";
import {
  FileBlob,
  PresentationFile,
} from "@oai/artifact-tool";

const workspace =
  "C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\.codex\\tmp\\final_selector_deck";
const starterPath = path.join(workspace, "template-starter.pptx");
const outputPath = path.join(workspace, "final-selector.pptx");
const renderDir = path.join(workspace, "final-render");
const layoutDir = path.join(workspace, "final-layout");

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function readImageBytes(filePath) {
  const bytes = await fs.readFile(filePath);
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
}

async function main() {
  const presentation = await PresentationFile.importPptx(
    await FileBlob.load(starterPath),
  );

  const before = await presentation.inspect({
    kind: "textbox,notes,chart,image",
    maxChars: 30000,
  });
  const records = before.ndjson
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  const findByText = (text, kind = "textbox") => {
    const record = records.find(
      (candidate) => candidate.kind === kind && candidate.text?.includes(text),
    );
    if (!record) {
      throw new Error(`編集対象が見つかりません: ${text}`);
    }
    return presentation.resolve(record.id);
  };

  const title = findByText(
    "Zhang 2019型（再現実装）は候補生成＋PAF内周選択",
  );
  title.text.replace(
    "Zhang 2019型（再現実装）は候補生成＋PAF内周選択",
    "Zhang 2019型（再現実装）は候補生成＋単純な最終選択",
  );

  const subtitle = findByText(
    "論文の処理で候補を作り、PAF固有条件で1つに絞る",
  );
  subtitle.text.replace(
    "論文の処理で候補を作り、PAF固有条件で1つに絞る",
    "内包関係があれば内側、なければ全周支持率最大",
  );

  const body = findByText(
    "② 最終選択（追加）：全周支持・明暗極性・同心候補から内周を採用",
  );
  body.text.replace(
    "② 最終選択（追加）：全周支持・明暗極性・同心候補から内周を採用",
    "② 最終選択（追加）：内包あり→内側／なし→全周支持率最大",
  );

  const notes = findByText(
    "次にPAF向けの追加処理として",
    "notes",
  );
  notes.setText(
    [
      "想定60秒。",
      "成果物では必ずZhang 2019型（再現実装）と表記する。",
      "著者コードの移植ではなく、論文の処理構成をPythonで再現したもの。",
      "まず論文と同様に、弧を分類・統合して候補楕円を作る。",
      "PAF向けの最終選択は単純な2規則とした。",
      "候補間に内包関係があれば最も内側を選び、内包関係がなければ全周支持率が最大の候補を選ぶ。",
      "図の緑は楕円近傍にCannyエッジがある部分、赤はない部分で、緑の割合が全周支持率。",
      "",
      "[Sources]",
      "- https://doi.org/10.3390/s19235243",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\paf_ring_detection\\methods\\zhang2019.py",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\paf_ring_detection\\methods\\zhang2019_paf.py",
    ].join("\n"),
  );

  findByText(
    "撮像診断：Zhang 2019型（再現実装）46.5%、CNN 74.5%",
  ).text.replace(
    "撮像診断：Zhang 2019型（再現実装）46.5%、CNN 74.5%",
    "撮像診断：Zhang 2019型（再現実装）47.6%、CNN 74.5%",
  );
  findByText("677枚＝46.5%／Canny 46.8%").text.replace(
    "677枚＝46.5%／Canny 46.8%",
    "693枚＝47.6%／Canny 46.8%",
  );
  findByText("CNNより28.0pt低い").text.replace(
    "CNNより28.0pt低い",
    "CNNより26.9pt低い",
  );
  const comparisonChartRecord = records.find(
    (record) => record.kind === "chart" && record.slide === 7,
  );
  if (!comparisonChartRecord) {
    throw new Error("第7枚の比較グラフが見つかりません");
  }
  const comparisonChart = presentation.resolve(comparisonChartRecord.id);
  comparisonChart.series.getItemAt(1).values = [47.6];

  findByText("黒つぶれ75%で14.3%").text.replace(
    "黒つぶれ75%で14.3%",
    "黒つぶれ75%で20.5%",
  );
  const severityImageRecord = records.find(
    (record) => record.kind === "image" && record.slide === 8,
  );
  if (!severityImageRecord) {
    throw new Error("第8枚の劣化別グラフ画像が見つかりません");
  }
  const severityImage = presentation.resolve(severityImageRecord.id);
  const oldFrame = severityImage.frame;
  const oldCrop = severityImage.crop;
  const oldFit = severityImage.fit;
  const oldGeometry = severityImage.geometry;
  const oldBorderRadius = severityImage.borderRadius;
  const oldRotation = severityImage.rotation;
  const oldFlipHorizontal = severityImage.flipHorizontal;
  const oldFlipVertical = severityImage.flipVertical;
  const oldLockAspectRatio = severityImage.lockAspectRatio;
  severityImage.replace({
    blob: await readImageBytes(
      "C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\diagnostic_by_severity.png",
    ),
    contentType: "image/png",
    alt: "新しい最終選択による撮像劣化強度別の検出成功率",
    ...(oldFit ? { fit: oldFit } : {}),
  });
  severityImage.frame = oldFrame;
  severityImage.crop = oldCrop;
  severityImage.geometry = oldGeometry;
  severityImage.borderRadius = oldBorderRadius;
  severityImage.rotation = oldRotation;
  severityImage.flipHorizontal = oldFlipHorizontal;
  severityImage.flipVertical = oldFlipVertical;
  severityImage.lockAspectRatio = oldLockAspectRatio;

  findByText("撮像診断で46.5%（Canny 46.8%、CNN 74.5%）").text.replace(
    "撮像診断で46.5%（Canny 46.8%、CNN 74.5%）",
    "撮像診断で47.6%（Canny 46.8%、CNN 74.5%）",
  );

  findByText(
    "撮像診断1,456枚では、Zhang 2019型（再現実装）は677枚成功で46.5%。",
    "notes",
  ).setText(
    [
      "想定55秒。",
      "撮像診断1,456枚では、Zhang 2019型（再現実装）は693枚成功で47.6%。",
      "Canny統制の46.8%とはほぼ同等。",
      "CNN方式は74.5%で、Zhang型より26.9ポイント高い。",
      "",
      "[Sources]",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\summary.json",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\diagnostic\\canny_contour_ransac.csv",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\diagnostic\\zhang2019.csv",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\diagnostic\\cnn_ransac.csv",
    ].join("\n"),
  );
  findByText(
    "黒つぶれも境界が背景へ埋まり、75%で14.3%、100%で8.9%。",
    "notes",
  ).setText(
    [
      "想定65秒。",
      "白飛びは明るくなっても境界勾配が残るため、Zhang型は強度100%でも83.9%。",
      "黒矩形は50%で3.6%、75%以上で0%。隠れた側の弧が不足し、三弧条件を作れない。",
      "黒つぶれも境界が背景へ埋まり、75%で20.5%、100%で8.9%。",
      "幾何拘束は誤候補を減らせる一方、必要な弧そのものが消える条件には弱い。",
      "",
      "[Sources]",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\diagnostic\\zhang2019.csv",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\diagnostic_by_severity.png",
    ].join("\n"),
  );
  findByText(
    "Zhang 2019型（再現実装）は46.5%で、Canny統制と同等、CNNより低かった。",
    "notes",
  ).setText(
    [
      "想定35秒。",
      "評価は撮像診断に一本化した。",
      "Zhang 2019型（再現実装）は47.6%で、Canny統制と同等、CNNより低かった。",
      "遮蔽や黒つぶれで弧自体が不足すると、三弧統合が成立しない。",
      "model_generalization_studyは今回の本編から外し、次回以降にCNNの未知条件一般化として扱う。",
      "",
      "[Sources]",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\.codex\\PROJECT_CONTEXT.md",
      "- C:\\Users\\siomi\\OneDrive\\デスクトップ\\Kenkyuu\\output\\results\\summary.json",
    ].join("\n"),
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
    maxChars: 30000,
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
