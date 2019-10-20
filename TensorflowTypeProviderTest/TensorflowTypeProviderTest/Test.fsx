#r "netstandard"
#r @"..\lib\System.Reflection.dll"
#r @"..\lib\System.Runtime.dll"
#r @"..\lib\Google.Protobuf.dll"
#r @"..\lib\TensorFlow.NET.dll"
#r @"..\lib\TensorflowTypeProvider.dll"

let [<Literal>] testData = __SOURCE_DIRECTORY__ + @"\..\..\TestData\"

open TensorflowTypeProvider

let [<Literal>] float_8x2Npy = testData + "float32_8x2.npy"
let [<Literal>] float_16Npy = testData + "float32_16.npy"
let [<Literal>] intNpy = testData + "int32.npy"

type NPY = TFProvider<float_16Npy>
let npy = NPY()
let v = npy.Values.[0] + 10.f



let [<Literal>] testDataNpz = testData + "test_data.npz"
type TestDataNPZ = TFProvider<testDataNpz>
let y = TestDataNPZ()
[for KeyValue(k,v) in y -> k]
[for KeyValue(k,v) in y.float32 -> k]
y.float32.shape_16.Values.[0]
y.float32.shape_8x2.Values.[0,0]

let [<Literal>] fashionTFGraph = testData + "FashionMNIST\FashionFrozen.pb"
type FashionTFGraph = TFProvider<fashionTFGraph>
let z = FashionTFGraph()
z.AllOps.Const.``conv2d/kernel``.AttrList.value
z.AllNodes.``conv2d/bias``.AttrList.value
let conv2d = z.AllNodes.``conv2d/Conv2D``
conv2d.AttrList.data_format
conv2d.AttrList.padding




