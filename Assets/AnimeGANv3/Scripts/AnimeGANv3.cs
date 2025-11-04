using System.Collections;
using System.Diagnostics;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;
using Debug = UnityEngine.Debug;

public class AnimeGANv3 : MonoBehaviour
{
    public RawImage rawImage;
    public Texture sourceTexture;
    public RenderTexture resultTexture;
    public ModelAsset modelAsset;
    private Worker _worker;
    private Model _runtimeModel;
    private Tensor<float> _inputTensor;
     
    void Awake()
    {
        // Load the model
        _runtimeModel = ModelLoader.Load(modelAsset);
        _worker = new Worker(_runtimeModel, BackendType.GPUCompute);

        _inputTensor = new Tensor<float>(new TensorShape(1, sourceTexture.width, sourceTexture.height, 3)); 
    }

    private void Start()
    {

    }

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            StartCoroutine(ProcessVideoMatting());
        }
    }

    IEnumerator ProcessVideoMatting()
    {

        if (sourceTexture == null)
        {
            yield break; 
        }

        int textureWidth = sourceTexture.width;
        int textureHeight = sourceTexture.height;

        var inputShape = new TensorShape(1, textureHeight, textureWidth, 3); // batch, height, width, channel
        _inputTensor?.Dispose();
        // 确保这里的通道数 C=3 (RGB) 与您的模型输入要求一致
        _inputTensor = new Tensor<float>(inputShape);

        TextureTransform textureTransform = new TextureTransform();
        textureTransform.SetDimensions(textureWidth, textureHeight, 3);
        textureTransform.SetChannelSwizzle(ChannelSwizzle.RGBA);
        textureTransform.SetTensorLayout(TensorLayout.NHWC);
        TextureConverter.ToTensor(sourceTexture, _inputTensor, textureTransform);
         
        _worker.Schedule(_inputTensor);

        yield return null;

        var outputTensor = _worker.PeekOutput() as Tensor<float>;

        var outputAwaiter = outputTensor.ReadbackAndCloneAsync().GetAwaiter();

        while (!outputAwaiter.IsCompleted)
        {
            yield return null;
        }

        using (var output = outputAwaiter.GetResult())
        {
            TextureTransform temp = new TextureTransform();
            temp.SetDimensions(textureWidth, textureHeight, 3);
            temp.SetTensorLayout(TensorLayout.NCHW);
            TextureConverter.RenderToTexture(outputTensor, resultTexture);
        }

        if (rawImage != null)
        {
            rawImage.texture = resultTexture;
        }
    }

    void OnDestroy()
    {
        _inputTensor?.Dispose();
        _worker?.Dispose();

    }
}