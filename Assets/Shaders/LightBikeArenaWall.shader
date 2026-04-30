Shader "LightBike/ArenaWall"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (0, 0.23, 0.56, 1)
        _EdgeColor ("Edge Color", Color) = (0.11, 0.25, 0.40, 1)
        _BaseEmission ("Base Emission", Range(0, 2)) = 0.06
        _EdgeEmission ("Edge Emission", Range(0, 2)) = 0.28
        _EdgePower ("Edge Power", Range(0.5, 8)) = 4.8
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
                float4 _BaseColor;
                float4 _EdgeColor;
                float _BaseEmission;
                float _EdgeEmission;
                float _EdgePower;
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
                float edge = pow(1.0 - ndv, max(0.01, _EdgePower));

                float3 baseColor = _BaseColor.rgb + (_BaseColor.rgb * _BaseEmission);
                float3 edgeGlow = _EdgeColor.rgb * (edge * _EdgeEmission);
                return half4(baseColor + edgeGlow, 1.0);
            }
            ENDHLSL
        }
    }
}
