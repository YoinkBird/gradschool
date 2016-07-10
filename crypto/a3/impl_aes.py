#!/usr/bin/env python3

from Crypto.Cipher import AES
# https://www.dlitz.net/software/pycrypto/api/current/Crypto.Util.number-module.html
from Crypto.Util.number import bytes_to_long,long_to_bytes

import binascii

# encode/decode: http://pythoncentral.io/encoding-and-decoding-strings-in-python-3-x/
if __name__ == "__main__":
    # 128-bit key k
    key = '00000000000000000000000000000001'
    #key = key.decode("hex")
    key = binascii.unhexlify(key)
    ptext = '80007000600050004000300020001000'
    ptext = binascii.unhexlify(ptext)
    aes=AES.new(key)
    ## ensure key is correct length
    # aes_obj.key_size=128
    ctext = aes.encrypt(ptext)
    ctext = binascii.hexlify(ctext)
    print("reference ctext:")
    print("9a7e0594961831b321efa7e06bdd4381")
    print("generated ctext:")
    print(bytes.decode(ctext))
    print("P:" + bytes.decode(binascii.hexlify(ptext)))
    print("K:" + bytes.decode(binascii.hexlify(key)))
    print("C:" + bytes.decode(ctext))

    print("")
    exit


'''
hex:
verify: http://aes.online-domain-tools.com/link/71ce93g08Piu2e7S4/
80007000600050004000300020001000
00000000000000000000000000000001
9a7e0594961831b321efa7e06bdd4381

ascii:
80007000600050004000300020001000
00000000000000000000000000000001
34ec503408d5b1090d351ab0171227aab2b786136b335c8a1bbf5b0cc16b76a7
# testing with long_to_bytes(bytes_to_long(cipher)).encode("hex")
34ec503408d5b1090d351ab0171227aab2b786136b335c8a1bbf5b0cc16b76a7
'''
