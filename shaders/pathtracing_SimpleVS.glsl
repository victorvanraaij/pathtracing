#version 330

layout (location = 0) in vec2 pos;

varying vec2 texCoord;

void main(void)
{
   //screen aligned quad
   gl_Position = vec4( pos.xy, 0.0, 1.0 );
   gl_Position = sign( gl_Position );
    
   // Texture coordinate for the quad
   texCoord = gl_Position.xy * 0.5 + 0.5; 
   
}