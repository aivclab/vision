async function runExample() {
  // Create an ONNX inference session with WebGL backend.
  const session = new onnx.InferenceSession({ backendHint: 'webgl' });

  await session.loadModel("./super_resolution.onnx");

  // load image.
  const imageLoader = new ImageLoader(imageSize, imageSize);
  const imageData = await imageLoader.getImageData('./cat.jpg');

  // preprocess the image data to match input dimension requirement, which is 1*3*224*224
  const width = imageSize;
  const height = imageSize;
  const preprocessedData = imageData.data;

  let y_channel = new Float32Array(width*height);
  for(let i = 0; i < preprocessedData.length/4; i++) {
    const a = create(preprocessedData[i*4],
        preprocessedData[i*4+1],
        preprocessedData[i*4+2]);
    y_channel[i] = a.y /255;
  }
  console.log(y_channel);

  const inputTensor = new onnx.Tensor(y_channel, 'float32', [1, 1, width, height]);
  // Run model with Tensor inputs and get the result.
  const outputMap = await session.run([inputTensor]);
  const outputData = outputMap.values().next().value.data;

  console.log(outputData);
  // Render the output result in html.
  //printMatches(outputData);

  const new_width = 3*width;
  const new_height= 3*height;

  const predictions = document.getElementById('predictions');
  predictions.width = new_width;
  predictions.height = new_height;
  var ctx = predictions.getContext("2d");
  const imageData2 = ctx.createImageData(new_width, new_height);

  // Iterate through every pixel
  for (let i = 0; i < imageData2.data.length; i += 4) {
    // Modify pixel data
    imageData2.data[i + 0] = outputData[i/4]*255;  // R value
    imageData2.data[i + 1] = outputData[i/4]*255;    // G value
    imageData2.data[i + 2] = outputData[i/4]*255;  // B value
    imageData2.data[i + 3] = 255;  // A value
  }

  // Draw image data to the canvas
  ctx.putImageData(imageData2, 0, 0);
}

function create(r, g, b) {
  return new ycbcr(r, g, b)
}

function ycbcr(r, g, b) {
  this.y  = ( .299 * r + .587 * g  +  0.114 * b) + 0
  this.cb = ( -.169 * r + -.331 * g +  0.500 * b) + 128
  this.cr = ( .500 * r + -.419 * g +  -0.081 * b) + 128
}

/**
 * Utility function to post-process SqueezeNet output. Find top k ImageNet classes with highest probability.
 */
function imagenetClassesTopK(classProbabilities, k) {
  if (!k) { k = 5; }
  const probs = Array.from(classProbabilities);
  const probsIndices = probs.map(
    function (prob, index) {
      return [prob, index];
    }
  );
  const sorted = probsIndices.sort(
    function (a,b) {
      if(a[0] < b[0]) {
        return -1;
      }
      if(a[0] > b[0]) {
        return 1;
      }
      return 0;
    }
  ).reverse();
  const topK = sorted.slice(0, k).map(function (probIndex) {
    const iClass = imagenetClasses[probIndex[1]];
    return {
      id: iClass[0],
      index: parseInt(probIndex[1], 10),
      name: iClass[1].replace(/_/g, ' '),
      probability: probIndex[0]
    };
  });
  return topK;
}

/**
 * Render SqueezeNet output to Html.
 */
function printMatches(data) {
  let outputClasses = [];
  if (!data || data.length === 0) {
    const empty = [];
    for (let i = 0; i < 5; i++) {
      empty.push({ name: '-', probability: 0, index: 0 });
    }
    outputClasses = empty;
  } else {
    outputClasses = imagenetClassesTopK(data, 5);
  }
  const predictions = document.getElementById('predictions');
  predictions.innerHTML = '';
  const results = [];
  for (let i of [0, 1, 2, 3, 4]) {
    results.push(`${outputClasses[i].name}: ${Math.round(100 * outputClasses[i].probability)}%`);
  }
  predictions.innerHTML = results.join('<br/>');
}
