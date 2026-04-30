Shader "LightBike/FloorGrid"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (0.02, 0.04, 0.08, 1)
        _LineColor ("Line Color", Color) = (0.11, 0.25, 0.40, 1)
        _CellSize ("Cell Size", Float) = 1
        _LineWidth ("Line Width", Range(0.001, 0.2)) = 0.03
        _Feather ("AA Feather", Range(0.2, 4)) = 2
        _LineEmission ("Line Emission", Range(0, 2)) = 0.23
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
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
            };

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
                float4 _LineColor;
                float _CellSize;
                float _LineWidth;
                float _Feather;
                float _LineEmission;
            CBUFFER_END

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                OUT.positionWS = TransformObjectToWorld(IN.positionOS.xyz);
                OUT.positionHCS = TransformWorldToHClip(OUT.positionWS);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float safeCellSize = max(_CellSize, 0.0001);
                float2 gridUV = IN.positionWS.xz / safeCellSize;

                // Bike/trail cell centers are integer world coordinates.
                // Keep grid lines on half-integers so visual boundaries match spawned trail cells.
                float2 centered = abs(frac(gridUV) - 0.5);
                float distToLine = min(centered.x, centered.y);

                float edgeAA = fwidth(distToLine) * max(_Feather, 0.2);
                float lineMask = 1.0 - smoothstep(_LineWidth, _LineWidth + edgeAA, distToLine);

                float3 baseColor = _BaseColor.rgb;
                float3 lineColor = _LineColor.rgb;
                float3 litColor = lerp(baseColor, lineColor, lineMask);
                float3 emissive = litColor + (lineColor * lineMask * _LineEmission);

                return half4(emissive, 1.0);
            }
            ENDHLSL
        }
    }
}
