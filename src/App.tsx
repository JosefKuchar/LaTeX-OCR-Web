import { Component, createEffect, createSignal, onMount } from "solid-js";
import { createStore } from "solid-js/store";
import katex from "katex";
import "jimp/browser/lib/jimp";
const { Jimp } = window as typeof window & { Jimp: any };
import config from "./config";
import { createDropzone } from "@solid-primitives/upload";
import {
  decode,
  detokenize,
  encode,
  normalize,
  postProcess,
  softmax,
  toGray,
} from "./utils";

const App: Component = () => {
  const [predicted, setPredicted] = createSignal("");
  const [preditecRender, setPredictedRender] = createSignal("");
  const [status, setStatus] = createSignal("");
  const [sessions, setSessions] = createStore({
    resizerSession: null,
    encSession: null,
    decSession: null,
  });
  let resultArea!: HTMLTextAreaElement;

  createEffect(async () => {
    setStatus("Loading models");
    const resizerSession = await ort.InferenceSession.create(
      "./image_resizer.onnx"
    );
    const encSession = await ort.InferenceSession.create("./encoder.onnx");
    const decSession = await ort.InferenceSession.create("./decoder.onnx");
    setSessions({
      resizerSession,
      encSession,
      decSession,
    });
    setStatus("Ready");
  });

  const handleFileSelect = async (event: any) => {
    const files = event.target.files;
    const reader = new FileReader();
    reader.onload = async (e: any) => {
      predictImg(e.target.result);
    };
    reader.readAsArrayBuffer(files[0]);
  };

  const fileInput = (
    <input type="file" class="hidden" onChange={handleFileSelect} />
  );

  const { setRef: dropzoneRef, files: droppedFiles } = createDropzone({
    onDrop: async (files: any) => {
      const reader = new FileReader();
      reader.onload = async (e: any) => {
        predictImg(e.target.result);
      };
      files.forEach((file: any) => {
        reader.readAsArrayBuffer(file.file);
      });
    },
  });

  document.onpaste = function (event) {
    var items = (
      event.clipboardData || (event as any).originalEvent.clipboardData
    ).items;
    for (const index in items) {
      var item = items[index];
      if (item.kind === "file") {
        var blob = item.getAsFile();
        var reader = new FileReader();
        reader.onload = function (event) {
          predictImg(event.target?.result as any);
        };
        reader.readAsArrayBuffer(blob);
      }
    }
  };

  const predictImg = async (imageData: ArrayBuffer) => {
    setStatus("Resizing image");

    // const canvas = document.createElement("canvas") as HTMLCanvasElement;
    // canvas.width = 250;
    // canvas.height = 122;
    // const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    // ctx?.drawImage(img as any, 0, 0, 250, 122);
    // const data = ctx?.getImageData(0, 0, 250, 122).data;
    // console.log(data);
    // const gray = to_gray(data);
    // console.log(gray);
    // const norm = normalize(gray);
    // console.log(norm);

    const image = await Jimp.read(imageData);
    image.resize(128, 64);
    const gray = toGray(image.bitmap.data);
    const norm = normalize(gray);
    // const resizerRes = await resizerSession.run({
    //   input: new ort.Tensor("float32", norm, [1, 1, 250, 122]),
    // });
    // const resizerIndex = resizerRes.output.data.indexOf(
    //   Math.max(...resizerRes.output.data)
    // );
    // const predictedWidth = (resizerIndex + 1) * 32;

    // console.log(predictedWidth);

    // Run encoder
    setStatus("Running encoder");
    const res: any = await encode(sessions.encSession, norm);

    // Prepare decoder input
    const out = [1n];
    const mask = [true];
    // Run decoder token by token
    setStatus("Running decoder ");
    for (let i = 0; i < config.max_seq_len; i++) {
      setStatus(status() + ".");
      const decRes: any = await decode(sessions.decSession, out, mask, res);
      // Get the last token logits
      const decOut = decRes.output.data;
      const logits = decOut.slice(decOut.length - decRes.output.dims[2]);
      const softmaxOut = softmax(logits);

      // Select the most probable character
      // TODO: Use random sampling
      const char = softmaxOut.indexOf(Math.max(...softmaxOut));
      out.push(BigInt(char));
      mask.push(true);

      // Stop if the character is EOS
      // TODO: remove hardcoded value
      if (char === 2) {
        break;
      }
    }

    const text = detokenize(out.map((x) => Number(x)));
    const postProcessed = postProcess(text);

    setPredicted(postProcessed);
    setPredictedRender(
      katex.renderToString(postProcessed, {
        output: "mathml",
      })
    );
    resultArea.select();
    resultArea.setSelectionRange(0, 99999);
  };

  const handleCopy = () => {
    resultArea.select();
    resultArea.setSelectionRange(0, 99999);
    navigator.clipboard.writeText(resultArea.value);
  };

  return (
    <div class="px-2 container mx-auto bg-slate-900 text-white">
      <div class="font-semibold text-3xl pt-5">LaTeX OCR Web</div>
      <div class="mb-2">
        <div>
          Made with 💖 by{" "}
          <a class="underline" href="https://josefkuchar.com">
            Josef Kuchař
          </a>
          ,{" "}
          <a
            class="underline"
            href="https://github.com/JosefKuchar/LaTeX-OCR-Web"
          >
            github.com/JosefKuchar/LaTeX-OCR-Web
          </a>
        </div>
        <div>
          Credits:{" "}
          <a
            class="underline"
            href="https://github.com/lukas-blecher/LaTeX-OCR/"
          >
            github.com/lukas-blecher/LaTeX-OCR
          </a>
          ,
          <a class="underline" href="https://github.com/RapidAI/RapidLatexOCR/">
            github.com/RapidAI/RapidLatexOCR
          </a>
        </div>
        <div>💡 You can directly paste image from clipboard</div>
        <div>⚠️ Work in progress</div>
      </div>
      <div>
        <span class="font-semibold mb-2">Status:</span> {status()}
      </div>
      <div class="mb-2">{fileInput}</div>
      <div
        ref={dropzoneRef}
        class="rounded border w-full h-20 p-2 text-center cursor-pointer hover:bg-slate-700 mb-2"
        onClick={() => (fileInput as any).click()}
      >
        Drop image here (or click to select)
      </div>
      <div class="font-semibold">Result</div>
      <textarea class="w-full bg-slate-900 border rounded" ref={resultArea}>
        {predicted()}
      </textarea>
      <button
        class="border px-2 py-1 hover:bg-slate-700 mb-2 rounded"
        onClick={handleCopy}
      >
        Copy
      </button>
      <div class="font-semibold">Preview</div>
      <div class="text-xl" innerHTML={preditecRender()} />
    </div>
  );
};

export default App;
