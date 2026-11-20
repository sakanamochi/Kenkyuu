import fs from 'node:fs/promises';
import {finalizePresentation} from 'file:///C:/Users/siomi/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations/container_tools/artifact_tool_utils.mjs';
const workspaceDir = 'C:/Users/siomi/OneDrive/デスクトップ/Kenkyuu';
const skillDir = 'C:/Users/siomi/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations';
const outputDir = `${workspaceDir}/研究関連書類/進捗報告会_202610/発表者ノート`;
await fs.mkdir(outputDir, {recursive: true});
// 元資料の19ページと寸法を保持したノート付き資料を検証する。
const result = await finalizePresentation({
  workspaceDir,
  candidatePath: `${workspaceDir}/.codex/notes_preview_20261026/candidate.pptx`,
  finalPath: `${outputDir}/ゼミ資料_発表者ノート付き.pptx`,
  explicitTotalSlideCount: 19,
  pythonExecutable: 'C:/Users/siomi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath: `${skillDir}/container_tools/inspect_presentation_package_integrity.py`,
  layoutValidatorPath: `${skillDir}/container_tools/inspect_presentation_layout_geometry.py`,
  layoutArgs: ['--expected-slide-size-emu','12192000,6858000','--require-native-table-slide','13'],
  requiredNativeTableOwnerSlides: [13],
  verifyArtifactToolImport: true,
  receiptPath: `${workspaceDir}/.codex/notes_preview_20261026/validation.json`,
});
console.log(JSON.stringify(result));
