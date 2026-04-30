Shader "LightBike/NeonBike"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (0, 0.94, 1, 1)
        _AccentColor ("Accent Color", Color) = (0, 0.23, 0.56, 1)
        _AccentStrength ("Accent Strength", Range(0, 1)) = 0.35
        _EmissionStrength ("Emission Strength", Range(0, 8)) = 2.5
        _FresnelPower ("Fresnel Power", Range(0.5, 8)) = 4.2
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
                float4 _AccentColor;
                float _AccentStrength;
                float _EmissionStrength;
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
                float fresnel = pow(1.0 - ndv, max(0.01, _FresnelPower));
                float accentMask = saturate(fresnel * _AccentStrength);

                float3 neonColor = lerp(_BaseColor.rgb, _AccentColor.rgb, accentMask);
                float3 emissive = neonColor * _EmissionStrength;
                return half4(emissive, 1.0);
            }
            ENDHLSL
        }
    }
}
