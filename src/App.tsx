import { Component, createEffect, createSignal, onMount } from "solid-js";
import katex from "katex";
import "jimp/browser/lib/jimp";
const { Jimp } = window as typeof window & { Jimp: any };
import config from "./config";
import { createDropzone } from "@solid-primitives/upload";
import { detokenize, normalize, postProcess, softmax, toGray } from "./utils";

const App: Component = () => {
  const [predicted, setPredicted] = createSignal("");
  const [preditecRender, setPredictedRender] = createSignal("");
  let status: any;

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
    const resizerSession = await ort.InferenceSession.create(
      "./image_resizer.onnx"
    );
    const encSession = await ort.InferenceSession.create("./encoder.onnx");
    const decSession = await ort.InferenceSession.create("./decoder.onnx");
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
    setTimeout(() => {
      status.innerText = "Running encoder";
    }, 0);
    const res = await encSession.run({
      input: new ort.Tensor("float32", norm, [1, 1, 64, 128]),
    });

    // Prepare decoder input
    const out = [1n];
    const mask = [true];
    // Run decoder token by token
    status.innerText = "Running decoder ";
    for (let i = 0; i < config.max_seq_len; i++) {
      const decRes = await decSession.run({
        x: new ort.Tensor("int64", out, [1, i + 1]),
        mask: new ort.Tensor("bool", mask, [1, i + 1]),
        context: new ort.Tensor("float32", res.output.data, res.output.dims),
      });
      setTimeout(() => {
        status.innerText += ".";
      }, 0);
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
    console.log(preditecRender());
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
        <div>⚠️ Work in progress</div>
      </div>
      <div
        ref={dropzoneRef}
        class="border w-full h-20 p-2 text-center cursor-pointer hover:bg-slate-700"
        onClick={() => (fileInput as any).click()}
      >
        Drop image here (or click to select)
      </div>
      {fileInput}
      <div ref={status} />
      <div>{predicted()}</div>
      <div innerHTML={preditecRender()} />
    </div>
  );
};

export default App;
