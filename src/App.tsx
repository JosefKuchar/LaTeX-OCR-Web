import { Component, createEffect, createSignal, onMount } from "solid-js";
import katex from "katex";
import "jimp/browser/lib/jimp";
const { Jimp } = window as typeof window & { Jimp: any };
import tokenizer from "./models/tokenizer.json";
import config from "./config";
import { createDropzone } from "@solid-primitives/upload";

const to_gray = (data: Uint8ClampedArray) => {
  const new_data = new Uint8ClampedArray(data.length / 4);
  for (let i = 0; i < data.length; i += 4) {
    new_data[i / 4] = (data[i] + data[i + 1] + data[i + 2]) / 3;
  }
  return new_data;
};

const normalize = (data: Uint8ClampedArray) => {
  const new_data = new Float32Array(data.length);
  let sum = 0;
  let max = 1;
  for (let i = 0; i < data.length; i++) {
    sum += data[i];
    if (data[i] > max) {
      max = data[i];
    }
  }

  //FIXME: Calculate mean and std
  const mean = 0.7931;
  const std = 0.1738;

  for (let i = 0; i < data.length; i++) {
    new_data[i] = (data[i] - mean * max) / (std * max);
  }
  return new_data;
};

const getOutput = (data: number[]) => {
  let string = "";

  for (let i = 0; i < data.length; i++) {
    const s = Object.entries(tokenizer.model.vocab).find(
      ([ke, val]) => val === data[i]
    );
    if (s) {
      string += s[0];
    } else {
      throw new Error("Not found");
    }
  }

  return string
    .replace(/\u0120/g, "")
    .replace("[EOS]", "")
    .replace("[BOS]", "")
    .replace("[PAD]", "")
    .trim();
};

const App: Component = () => {
  const [predicted, setPredicted] = createSignal("");
  const [preditecRender, setPredictedRender] = createSignal("");

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
      "./src/models/image_resizer.onnx"
    );

    const encSession = await ort.InferenceSession.create(
      "./src/models/encoder.onnx"
    );
    const decSession = await ort.InferenceSession.create(
      "./src/models/decoder.onnx"
    );
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
    const gray = to_gray(image.bitmap.data);
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
    const res = await encSession.run({
      input: new ort.Tensor("float32", norm, [1, 1, 64, 128]),
    });

    // Prepare decoder input
    const out = [1n];
    const mask = [true];
    // Run decoder token by token
    for (let i = 0; i < config.max_seq_len; i++) {
      const decRes = await decSession.run({
        x: new ort.Tensor("int64", out, [1, i + 1]),
        mask: new ort.Tensor("bool", mask, [1, i + 1]),
        context: new ort.Tensor("float32", res.output.data, res.output.dims),
      });
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

    setPredicted(getOutput(out.map((x) => Number(x))));
    setPredictedRender(
      katex.renderToString(predicted(), {
        output: "mathml",
      })
    );
    console.log(preditecRender());
  };

  return (
    <>
      <div
        ref={dropzoneRef}
        style={{ width: "100px", height: "100px", background: "red" }}
      >
        Dropzone
      </div>
      <div>{predicted()}</div>
      <div innerHTML={preditecRender()} />
    </>
  );
};

export default App;

function softmax(input: number[]) {
  // Calculate the maximum value in the input array to avoid numerical instability
  const maxVal = Math.max(...input);
  // Calculate the sum of the exponentials of the input array elements
  const expSum = input.reduce((acc, val) => acc + Math.exp(val - maxVal), 0);
  // Apply the softmax transformation to each element of the input array
  const softmaxOutput = input.map((val) => Math.exp(val - maxVal) / expSum);
  return softmaxOutput;
}
