import { Component, onMount } from "solid-js";

import logo from "./logo.svg";
import styles from "./App.module.css";
import tokenizer from "./tokenizer";

const to_gray = (data: Uint8ClampedArray) => {
  const new_data = new Uint8ClampedArray(data.length / 4);
  for (let i = 0; i < data.length; i += 4) {
    new_data[i / 4] =
      //0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];

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

const out = [
  61, 205, 105, 103, 141, 102, 106, 103, 180, 102, 311, 104, 69, 128, 105, 103,
  122, 102, 106, 103, 107, 102, 104, 69, 128, 105, 103, 122, 102, 106, 103, 107,
  102, 104, 69, 150, 105, 103, 122, 102, 106, 103, 107, 102, 104, 139, 128, 105,
  103, 122, 102, 106, 103, 148, 102, 104, 69, 150, 105, 103, 122, 102, 106, 103,
  107, 102, 104, 69, 150, 105, 103, 122, 102, 106, 103, 148, 102, 104, 327, 104,
  327, 104, 327, 104, 327, 104, 327, 2,
];

const App: Component = () => {
  const img = <img src="/src/HAbJw.png" />;

  onMount(async () => {
    // const session = await ort.InferenceSession.create("./src/model.onnx");

    // const dataA = new Float32Array(new Array(112 * 464).fill(0.1));

    // const input = {
    //   input_image: new ort.Tensor("float32", dataA, [1, 1, 112, 464]),
    // };

    // console.log(input);

    // const result = await session.run(input);
    setTimeout(async () => {
      const canvas = document.createElement("canvas") as HTMLCanvasElement;
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
      ctx?.drawImage(img as any, 0, 0, 112, 464);
      const data = ctx?.getImageData(0, 0, 112, 464).data;
      console.log(data);
      const gray = to_gray(data);
      console.log(gray);
      const norm = normalize(gray);
      console.log(norm);

      const session = await ort.InferenceSession.create("./src/model.onnx");

      const input = {
        input_image: new ort.Tensor("float32", norm, [1, 1, 112, 464]),
      };

      const result = await session.run(input);
      console.log(result);

      console.log(getOutput(out));
    }, 1000);
  });

  return img;
};

export default App;
