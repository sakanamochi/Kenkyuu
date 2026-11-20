import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const root='C:/Users/siomi/OneDrive/デスクトップ/Kenkyuu';
const build=root+'/.codex/pptx_build_20261025';
const skill='C:/Users/siomi/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
// 編集セッションの確定データからPPTXを書き出す。
const data=JSON.parse(await fs.readFile(build+'/deck.json','utf8'));
// JSON保存された画像バイト列を復元し、画像をPPTX内部へ格納する。
for(const image of data.images) image.data=Uint8Array.from(Object.values(image.data));
const p=Presentation.load(data);
await(await PresentationFile.exportPptx(p)).save(build+'/candidate.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:build+'/candidate.pptx',finalPath:root+'/研究関連書類/進捗報告会_202610/進捗報告_伊藤弘道_発表用.pptx',
 pythonExecutable:'C:/Users/siomi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...([13,21,22].flatMap(n=>['--require-native-table-slide',String(n)]))],
 requiredNativeChartOwnerSlides:[10,14,15,16,17,18,19,23],requiredNativeTableOwnerSlides:[13,21,22],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Meiryo']},verifyArtifactToolImport:true,
 receiptPath:build+'/validation_final.json'});
console.log(JSON.stringify(result));
