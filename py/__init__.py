from .nodes import Text2VideoNode, ImageGeneratorNode, Image2VideoNode, KLingAIAPIClient, PreviewVideo, KolorsVirtualTryOnNode, VideoExtendNode, LipSyncNode, LipSyncTextInputNode, LipSyncAudioInputNode, EffectNode

NODE_CLASS_MAPPINGS = {
    'Client': KLingAIAPIClient,
    'Image Generator': ImageGeneratorNode,
    'Text2Video': Text2VideoNode,
    'Image2Video': Image2VideoNode,
    'Virtual Try On': KolorsVirtualTryOnNode,
    'KLingAI Preview Video': PreviewVideo,
    'Video Extend': VideoExtendNode,
    'Lip Sync': LipSyncNode,
    'Lip Sync Text Input': LipSyncTextInputNode,
    'Lip Sync Audio Input': LipSyncAudioInputNode,
    'Effects': EffectNode
}