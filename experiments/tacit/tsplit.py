import sys
FN=["FTb","FNt","FUj","FIj","FTrap","FSync","FRes1","FRes2"]
CN=["CTb","CNt","CNa","CIj"]
def varint(d,i):
    sc=[]
    while True:
        b=d[i]; i+=1; sc.append(b)
        if b&0x80: break
    v=0
    for b in reversed(sc): v=(v<<7)|(b&0x7f)
    return v,i
def split(d, first_is_sync=True):
    """yield (offset, length, kind, fields)"""
    i=0; first=True
    while i < len(d):
        st=i; b=d[i]; i+=1
        c=b&3
        if c!=2:
            yield (st,i-st,"C"+CN[c],{"ts":(b&0xfc)>>2}); first=False; continue
        f=(b>>2)&7; name=FN[f]; fl={}
        if f in (0,1,3):
            fl["ts"],i=varint(d,i)
        elif f==2:
            fl["tgt"],i=varint(d,i); fl["ts"],i=varint(d,i)
        elif f==5:
            if not first:
                fl["brmode"],i=varint(d,i)
            fl["tgt"],i=varint(d,i); fl["ts"],i=varint(d,i)
        elif f==4:
            fl["from"],i=varint(d,i); fl["tgt"],i=varint(d,i); fl["ts"],i=varint(d,i)
        else:
            print(f"  [tail] bad fheader {f} at {st}; stopping ({len(d)-st} bytes left)"); return
        yield (st,i-st,"F"+name,fl); first=False
if __name__=="__main__":
    d=open(sys.argv[1],'rb').read()
    n=int(sys.argv[2]) if len(sys.argv)>2 else 30
    pk=list(split(d))
    print("total packets:", len(pk), "bytes:", len(d))
    from collections import Counter
    print("kinds:", Counter(k for _,_,k,_ in pk).most_common())
    for off,ln,k,fl in pk[:n]:
        ex = ""
        if "tgt" in fl: ex=f" tgt=0x{fl['tgt']:x} (pc^ = 0x{fl['tgt']<<1:x})"
        print(f"  off={off:6d} len={ln} {k:8s} ts={fl.get('ts')}{ex}")
