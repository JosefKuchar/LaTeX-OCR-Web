import { Component, Show, createEffect, createSignal, onMount } from "solid-js";
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
  getData,
  normalize,
  postProcess,
  predictWidth,
  softmax,
  toGray,
} from "./utils";

const App: Component = () => {
  const [predicted, setPredicted] = createSignal("");
  const [predictedRender, setPredictedRender] = createSignal("");
  const [status, setStatus] = createSignal("");
  const [running, setRunning] = createSignal(false);
  const [cancel, setCancel] = createSignal(false);
  const [loaded, setLoaded] = createSignal(false);
  const [sessions, setSessions] = createStore({
    resizerSession: null,
    encSession: null,
    decSession: null,
  });
  let resultArea!: HTMLTextAreaElement;

  createEffect(async () => {
    setStatus("Downloading resizer model (38MB)");
    const resizerModel = await getData("./image_resizer.onnx");
    setStatus("Downloading encoder model (87MB)");
    const encModel = await getData("./encoder.onnx");
    setStatus("Downloading decoder model (50MB)");
    const decModel = await getData("./decoder.onnx");
    setStatus("Loading resizer model");
    const resizerSession = await ort.InferenceSession.create(resizerModel);
    setStatus("Loading encoder model");
    const encSession = await ort.InferenceSession.create(encModel);
    setStatus("Loading decoder model");
    const decSession = await ort.InferenceSession.create(decModel);
    setSessions({
      resizerSession,
      encSession,
      decSession,
    });
    setStatus("Ready");
    setLoaded(true);
  });

  const handleFileSelect = async (event: any) => {
    const files = event.target.files;
    const reader = new FileReader();
    reader.onload = async (e: any) => {
      predictImg(e.target.result);
    };
    if (files.length > 0) {
      reader.readAsArrayBuffer(files[0]);
    }
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
    if (loaded()) {
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
    }
  };

  const stopCalculations = () => {
    setRunning(false);
    setCancel(false);
    setStatus("Ready");
  };

  const predictImg = async (imageData: ArrayBuffer) => {
    setRunning(true);
    setStatus("Resizing image ");
    let image;
    try {
      image = await Jimp.read(imageData);
    } catch (e) {
      stopCalculations();
      setStatus("Error: Invalid image. Ready");
      return;
    }

    // Resize image to fit model input
    image.background(0xffffffff);
    if (image.bitmap.width > config.max_width) {
      image.resize(config.max_width, Jimp.AUTO);
    }
    if (image.bitmap.height > config.max_height) {
      image.resize(Jimp.AUTO, config.max_height);
    }

    // Contain to x32 size
    image.contain(
      Math.ceil(image.bitmap.width / 32) * 32,
      Math.ceil(image.bitmap.height / 32) * 32,
      Jimp.HORIZONTAL_ALIGN_LEFT | Jimp.VERTICAL_ALIGN_TOP
    );

    // Run resizer to find optimal width
    let width = image.bitmap.width;
    let height = image.bitmap.height;
    for (let i = 0; i < 10; i++) {
      setStatus(status() + ".");
      const gray = toGray(image.bitmap.data);
      const norm = normalize(gray);
      let predictedWidth: any;
      try {
        predictedWidth = await predictWidth(
          sessions.resizerSession,
          norm,
          width,
          height
        );
      } catch (e) {
        stopCalculations();
        setStatus("Error: Resizer error. Try again");
        return;
      }

      // Stop if the predicted width is bigger than the original (or the same)
      if (predictedWidth >= width) {
        break;
      }

      // Calculate height based on the predicted width (~ keep aspect ratio)
      const predictedHeight = Math.max(
        Math.round(
          ((predictedWidth / image.bitmap.width) * image.bitmap.height) / 32
        ) * 32,
        32
      );

      // Resize image
      image.resize(predictedWidth, predictedHeight);
      width = predictedWidth;
      height = predictedHeight;

      if (cancel()) {
        stopCalculations();
        return;
      }
    }

    const gray = toGray(image.bitmap.data);
    const norm = normalize(gray);

    if (cancel()) {
      stopCalculations();
      return;
    }

    // Run encoder
    setStatus("Running encoder ");
    let res;
    try {
      res = await encode(sessions.encSession, norm, width, height);
    } catch (e) {
      stopCalculations();
      setStatus("Error: Encoder error. Try again");
      return;
    }

    if (cancel()) {
      stopCalculations();
      return;
    }

    // Prepare decoder input
    const out = [BigInt(config.bos_token)];
    const mask = [true];

    // Run decoder token by token
    setStatus("Running decoder ");
    for (let i = 0; i < config.max_seq_len; i++) {
      setStatus(status() + ".");
      let decRes: any;
      try {
        decRes = await decode(sessions.decSession, out, mask, res);
      } catch (e) {
        stopCalculations();
        setStatus("Error: Decoder error. Try again");
        return;
      }
      // Get the last token logits
      const decOut = decRes.output.data;
      const logits = decOut.slice(decOut.length - decRes.output.dims[2]);
      const softmaxOut = softmax(logits);

      // Select the most probable character
      // TODO: Don't use greedy algorithm
      const char = softmaxOut.indexOf(Math.max(...softmaxOut));
      out.push(BigInt(char));
      mask.push(true);

      // Stop if the character is EOS
      if (char === config.eos_token) {
        break;
      }

      if (cancel()) {
        stopCalculations();
        return;
      }
    }

    const text = detokenize(out.map((x) => Number(x)));
    const postProcessed = postProcess(text);

    setPredicted(postProcessed);
    try {
      const rendered = katex.renderToString(postProcessed, {
        output: "mathml",
      });
      setPredictedRender(rendered);
    } catch {
      setPredictedRender(
        "Unable to render preview. Output is malformed, you may be able to fix it manually."
      );
    }
    resultArea.select();
    resultArea.setSelectionRange(0, 99999);
    stopCalculations();
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
      <div class="flex items-center">
        <div>
          <span class="font-semibold mb-2">Status:</span> {status()}
        </div>
        <Show when={running()}>
          <div class="ml-auto">
            <button
              class="border px-2 py-1 hover:bg-slate-700 mb-2 rounded"
              onClick={() => {
                setCancel(true);
              }}
            >
              Cancel
            </button>
          </div>
        </Show>
      </div>
      <Show when={loaded()}>
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
        <div class="text-xl" innerHTML={predictedRender()} />
      </Show>
    </div>
  );
};

export default App;
