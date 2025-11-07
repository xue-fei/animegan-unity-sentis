using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;

public class Test : MonoBehaviour
{
    public int webcamWidth = 640;
    public int webcamHeight = 480;
    private WebCamTexture webCamTexture;

    public ModelAsset modelAsset;
    private Worker _worker;
    private Model _runtimeModel;
    private Tensor<float> _inputTensor;
    private Tensor<float> tempTensor;
    private Tensor<float> outputTensor;
    TextureTransform textureTransform;
    public RawImage rawImage;
    public RenderTexture renderTexture;

    private void Awake()
    {
        textureTransform = new TextureTransform();
        textureTransform.SetDimensions(webcamWidth, webcamHeight, 3);
        textureTransform.SetTensorLayout(TensorLayout.NHWC);

        _runtimeModel = ModelLoader.Load(modelAsset);
        _worker = new Worker(_runtimeModel, BackendType.GPUCompute);
        _inputTensor = new Tensor<float>(new TensorShape(1, webcamWidth, webcamHeight, 3));

        InitializeWebcam();
    }

    void InitializeWebcam()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length > 0)
        {
            webCamTexture = new WebCamTexture(devices[0].name, webcamWidth, webcamHeight, 30);
            webCamTexture.Play();
            Debug.Log($"启动摄像头: {devices[0].name}");
        }
        else
        {
            Debug.LogWarning("未找到摄像头设备");
        }
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        ProcessFrame(WebCamTextureToTexture2D(webCamTexture));
    }

    void ProcessFrame(Texture2D sourceTexture)
    {
        TextureConverter.ToTensor(sourceTexture, _inputTensor, textureTransform);
        _worker.Schedule(_inputTensor);
        tempTensor = _worker.PeekOutput() as Tensor<float>;
        outputTensor = tempTensor.ReadbackAndClone();
        TextureConverter.RenderToTexture(outputTensor, renderTexture, textureTransform);
        if (rawImage != null)
        {
            rawImage.texture = renderTexture;
        }
    }

    Texture2D WebCamTextureToTexture2D(WebCamTexture webCamTexture)
    {
        Texture2D tex = new Texture2D(webCamTexture.width, webCamTexture.height, TextureFormat.RGBA32, false);
        tex.SetPixels32(webCamTexture.GetPixels32());
        tex.Apply();
        return tex;
    }

    void OnDestroy()
    {
        _inputTensor?.Dispose();
        _worker?.Dispose();
    }
}