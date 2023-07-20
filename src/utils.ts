import tokenizer from "./tokenizer.json";

/**
 * Convert RGBA data to grayscale
 * @param data RGBA data
 * @returns Grayscale data (1 channel)
 */
export const toGray = (data: Uint8ClampedArray) => {
  const new_data = new Uint8ClampedArray(data.length / 4);
  for (let i = 0; i < data.length; i += 4) {
    new_data[i / 4] = (data[i] + data[i + 1] + data[i + 2]) / 3;
  }
  return new_data;
};

/**
 * Normalize data
 * @param data Data to normalize
 * @returns Normalized data
 */
export const normalize = (data: Uint8ClampedArray) => {
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

/**
 * Convert tokenized data to string
 * @param data Tokenized data
 * @returns String
 */
export const detokenize = (data: number[]) => {
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
    .replace(/\u0120/g, " ")
    .replace("[EOS]", "")
    .replace("[BOS]", "")
    .replace("[PAD]", "")
    .trim();
};

/**
 * Remove unnecessary spaces
 * @param s String to postprocess
 * @returns Postprocessed string
 */
export const postProcess = (s: string) => {
  const textReg = /(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})/g;
  let names = [...s.matchAll(textReg)].map((x) => x[0].replace(" ", ""));
  s = s.replace(textReg, (match) => (names as any).shift());
  let news = s;
  while (true) {
    s = news;
    news = s.replace(/(?!\\ )([\W_^\d])\s+?([\W_^\d])/g, "$1$2");
    news = news.replace(/(?!\\ )([\W_^\d])\s+?([a-zA-Z])/g, "$1$2");
    news = news.replace(/([a-zA-Z])\s+?([\W_^\d])/g, "$1$2");
    if (news === s) {
      break;
    }
  }
  return s;
};

/**
 * Calculate softmax of an array
 * @param input Input array
 * @returns Softmax of the input array
 */
export const softmax = (input: number[]) => {
  // Calculate the maximum value in the input array to avoid numerical instability
  const maxVal = Math.max(...input);
  // Calculate the sum of the exponentials of the input array elements
  const expSum = input.reduce((acc, val) => acc + Math.exp(val - maxVal), 0);
  // Apply the softmax transformation to each element of the input array
  const softmaxOutput = input.map((val) => Math.exp(val - maxVal) / expSum);
  return softmaxOutput;
};

/**
 * Encode the input image using the encoder model
 * @param session Encoder model session
 * @param input Input image
 * @returns Encoded image context
 */
export const encode = (
  session: any,
  input: any,
  width: number,
  height: number
) => {
  return new Promise((resolve, reject) => {
    setTimeout(async () => {
      const res = await session.run({
        input: new ort.Tensor("float32", input, [1, 1, height, width]),
      });
      resolve(res);
    }, 0);
  });
};

/**
 * Decode one token using the decoder model
 * @param session Decoder model session
 * @param out Previous decoded tokens
 * @param mask Mask of the previous decoded tokens
 * @param context Encoded image context
 * @returns Decoded token (raw probability distribution)
 */
export const decode = (session: any, out: any, mask: any, context: any) => {
  return new Promise((resolve, reject) => {
    setTimeout(async () => {
      const res = await session.run({
        x: new ort.Tensor("int64", out, [1, out.length]),
        mask: new ort.Tensor("bool", mask, [1, mask.length]),
        context: new ort.Tensor(
          "float32",
          context.output.data,
          context.output.dims
        ),
      });
      resolve(res);
    }, 0);
  });
};

/**
 * Predict correct width of the image for encoder model
 * @param session Resizer model session
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @returns Predicted width
 */
export const predictWidth = (
  session: any,
  input: any,
  width: number,
  height: number
) => {
  return new Promise((resolve, reject) => {
    setTimeout(async () => {
      const resizerRes = await session.run({
        input: new ort.Tensor("float32", input, [1, 1, height, width]),
      });
      const resizerIndex = resizerRes.output.data.indexOf(
        Math.max(...resizerRes.output.data)
      );
      const predictedWidth = (resizerIndex + 1) * 32;
      resolve(predictedWidth);
    }, 0);
  });
};

// Get data from the cache.
async function getCachedData(cacheName: string, url: string) {
  const cacheStorage = await caches.open(cacheName);
  const cachedResponse = await cacheStorage.match(url);

  if (!cachedResponse || !cachedResponse.ok) {
    return false;
  }

  return await cachedResponse.arrayBuffer();
}

// Delete any old caches to respect user's disk space.
async function deleteOldCaches(currentCache: string) {
  const keys = await caches.keys();

  for (const key of keys) {
    const isOurCache = key.startsWith("myapp-");
    if (currentCache === key || !isOurCache) {
      continue;
    }
    caches.delete(key);
  }
}

export async function getData(url: string) {
  const cacheVersion = 1;
  const cacheName = `myapp-${cacheVersion}`;
  let cachedData = await getCachedData(cacheName, url);

  if (cachedData) {
    console.log("Retrieved cached data");
    return cachedData;
  }

  console.log("Fetching fresh data");

  const cacheStorage = await caches.open(cacheName);
  await cacheStorage.add(url);
  cachedData = await getCachedData(cacheName, url);
  await deleteOldCaches(cacheName);

  return cachedData;
}
