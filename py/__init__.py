from .nodes import Text2VideoNode, ImageGeneratorNode, TextToAudioNode, Video2AudioNode, ImageExpanderNode, \
    Image2VideoNode, KLingAIAPIClient, PreviewVideo, PreviewAudio, KolorsVirtualTryOnNode, VideoExtendNode, LipSyncNode, \
    LipSyncTextInputNode, LipSyncAudioInputNode, EffectNode

NODE_CLASS_MAPPINGS = {
    'Client': KLingAIAPIClient,
    'Image Generator': ImageGeneratorNode,
    'Image Expander': ImageExpanderNode,
    "Text2Audio": TextToAudioNode,
    "Video2Audio": Video2AudioNode,
    'Text2Video': Text2VideoNode,
    'Image2Video': Image2VideoNode,
    'Virtual Try On': KolorsVirtualTryOnNode,
    'KLingAI Preview Video': PreviewVideo,
    'KLingAI Preview Audio': PreviewAudio,
    'Video Extender': VideoExtendNode,
    'Lip Sync': LipSyncNode,
    'Lip Sync Text Input': LipSyncTextInputNode,
    'Lip Sync Audio Input': LipSyncAudioInputNode,
    'Effects': EffectNode
}
