"""Compare C++ wire CRC with the official Python SDK's Go pack formats."""
import os
import struct
import subprocess
import unittest


def crc(data):
  result=0xFFFFFFFF
  for (word,) in struct.iter_unpack('<I',data[:-4]):
    for bit in range(31,-1,-1):
      result=((result<<1)^ (0x04C11DB7 if result&0x80000000 else 0))&0xFFFFFFFF
      if word&(1<<bit): result^=0x04C11DB7
  return result


class WireCrcTest(unittest.TestCase):
  @unittest.skipUnless(os.environ.get('GO2W_CRC_TEST'),'Set GO2W_CRC_TEST to the compiled C++ fixture')
  def test_cpp_matches_python_protocol_and_ignores_padding(self):
    header=[254,239,255,12,1,2,3,4,500]
    command=header[:]
    for i in range(20): command += [1,struct.unpack('<f',struct.pack('<f',.01))[0]*i,2*i,-.5*i,20+i,.5,i+100,200,300]
    command += [1,5,6,7]+list(range(40))+list(range(10,22))+[1,2,3,42,0]
    cmd=struct.pack('<4B4IH2x'+'B3x5f3I'*20+'4B55Bx2I',*command)
    state=header+[1,.1,.2,.3,.4,.5,.6,1,2,3,.7,.8,.9,40]
    for i in range(20): state += [1,struct.unpack('<f',struct.pack('<f',.01))[0]*i,2*i,3*i,-.5*i,4*i,5*i,6*i,20+i,100+i,i,44]
    state += [1,2,3,4,-123,12,8,9,10,11]+list(range(12,27))
    state += [10,20,30,40,11,22,33,44,12345]+list(range(40))+[1,1.25,20,21,24.5,1.5,1,2,3,4,99,0]
    packed=struct.pack('<4B4IH2x13fb3x'+'B3x7fb3x3I'*20+'4BiH4b15H8hI41B3xf2b2x2f4h2I',*state)
    result=subprocess.run([os.environ['GO2W_CRC_TEST']],capture_output=True,text=True,check=True)
    self.assertEqual(list(map(int,result.stdout.split())),[crc(cmd),crc(packed)])


if __name__=='__main__': unittest.main()
