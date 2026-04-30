Shader "LightBike/NeonTrail"
{
    Properties
    {
        _GlowColor ("Glow Color", Color) = (0, 0.94, 1, 1)
        _CoreStrength ("Core Strength", Range(0, 8)) = 3.8
        _EdgeStrength ("Edge Strength", Range(0, 8)) = 1.9
        _FresnelPower ("Fresnel Power", Range(0.5, 8)) = 3.1
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "Queue" = "Geometry"
            "RenderPipeline" = "UniversalRenderPipeline"
        }

        Pass
        {
            Name "ForwardUnlit"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 normalWS : TEXCOORD0;
                float3 viewDirWS : TEXCOORD1;
            };

            CBUFFER_START(UnityPerMaterial)
                float4 _GlowColor;
                float _CoreStrength;
                float _EdgeStrength;
                float _FresnelPower;
            CBUFFER_END

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                float3 positionWS = TransformObjectToWorld(IN.positionOS.xyz);
                OUT.positionHCS = TransformWorldToHClip(positionWS);
                OUT.normalWS = TransformObjectToWorldNormal(IN.normalOS);
                OUT.viewDirWS = GetWorldSpaceNormalizeViewDir(positionWS);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float3 normalWS = normalize(IN.normalWS);
                float3 viewDirWS = normalize(IN.viewDirWS);
                float ndv = saturate(dot(normalWS, viewDirWS));
                float rim = pow(1.0 - ndv, max(0.01, _FresnelPower));

                float glowStrength = _CoreStrength + rim * _EdgeStrength;
                float3 emissive = _GlowColor.rgb * glowStrength;
                return half4(emissive, 1.0);
            }
            ENDHLSL
        }
    }
}
