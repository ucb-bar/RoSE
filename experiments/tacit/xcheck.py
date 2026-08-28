import re, sys, collections
sys.path.insert(0, "/tmp/claude-1172/-scratch-dima-rose-infra-RoSE/380445be-ecd7-4b3a-9531-bd33760b300b/scratchpad")

BR = {"beq","bne","blt","bge","bltu","bgeu","beqz","bnez","blez","bgez","bltz","bgtz","bgt","ble","bgtu","bleu"}
DJ = {"j","jal"}                       # direct / inferable
IJ = {"jr","jalr","ret","mret","sret","uret","tail"}   # indirect / uninferable

def load_dis(path):
    """addr -> (len, kind, target)  kind in n(ormal) b(ranch) d(irect jump) i(ndirect)"""
    insns = {}
    pat = re.compile(r"^\s+([0-9a-f]+):\s+([0-9a-f ]+?)\s+(\S+)\s*(.*)$")
    for line in open(path):
        m = pat.match(line)
        if not m: continue
        addr = int(m.group(1), 16)
        hexb = m.group(2).replace(" ", "")
        ln = len(hexb)//2
        if ln not in (2,4): continue
        mn = m.group(3); ops = m.group(4)
        kind, tgt = "n", None
        base = mn.split(".")[-1] if mn.startswith("c.") else mn
        if base in BR:
            kind = "b"
            mt = re.search(r"([0-9a-f]{6,})\s*<", ops)
            if mt: tgt = int(mt.group(1), 16)
        elif base in DJ:
            # `jal ra,<t>` and `j <t>` are direct; `jalr`-style handled below
            kind = "d"
            mt = re.search(r"([0-9a-f]{6,})\s*<", ops)
            if mt: tgt = int(mt.group(1), 16)
        elif base in IJ:
            kind = "i"
        insns[addr] = (ln, kind, tgt)
    return insns

def varint(d,i):
    sc=[]
    while True:
        b=d[i]; i+=1; sc.append(b)
        if b&0x80: break
        if len(sc)>10: raise ValueError("varint>10")
    v=0
    for b in reversed(sc): v=(v<<7)|(b&0x7f)
    return v,i

def run(elf_dis, trace, start_pc, skip_packets=0, maxreport=12):
    ins = load_dis(elf_dis)
    d = open(trace,'rb').read()
    # first packet = sync (6 bytes here)
    i = 0
    b=d[0]; i=1
    _,i = varint(d,i)   # target
    _,i = varint(d,i)   # ts
    pc = start_pc
    for _ in range(skip_packets):
        b=d[i]; i+=1
        if (b&3)==2:
            f=(b>>2)&7
            if f in (0,1,3): _,i=varint(d,i)
            elif f==2: _,i=varint(d,i); _,i=varint(d,i)
            elif f==4: _,i=varint(d,i); _,i=varint(d,i); _,i=varint(d,i)
    matched=0; reports=[]
    while i < len(d):
        # walk to next control-flow insn
        steps=0
        while True:
            e = ins.get(pc)
            if e is None:
                reports.append(f"  pkt#{matched}: PC {pc:#x} not an instruction")
                return matched, reports
            ln,kind,tgt = e
            if kind != "n": break
            pc += ln; steps += 1
            if steps > 100000:
                reports.append(f"  pkt#{matched}: runaway from {pc:#x}"); return matched, reports
        ln,kind,tgt = ins[pc]
        b = d[i]; i += 1
        c = b & 3
        if c != 2:
            ptype = {0:"Tb",1:"Nt",3:"Ij"}[c]
        else:
            f = (b>>2)&7
            ptype = {0:"Tb",1:"Nt",2:"Uj",3:"Ij",4:"Trap",5:"Sync"}.get(f,"?%d"%f)
            try:
                if f in (0,1,3): _,i=varint(d,i)
                elif f==2: t,i=varint(d,i); _,i=varint(d,i)
                elif f==4: _,i=varint(d,i); _,i=varint(d,i); _,i=varint(d,i)
                elif f==5: _,i=varint(d,i); _,i=varint(d,i); _,i=varint(d,i)
                else: raise ValueError("fheader %d"%f)
            except Exception as ex:
                reports.append(f"  pkt#{matched}: {ex} at byte {i}"); return matched, reports
        ok = ((kind=="b" and ptype in ("Tb","Nt")) or
              (kind=="d" and ptype=="Ij") or
              (kind=="i" and ptype=="Uj"))
        if not ok:
            if len(reports) < maxreport:
                reports.append(f"  pkt#{matched} byte@{i-1}: insn@{pc:#x} kind={kind} but packet={ptype}")
            return matched, reports
        # advance
        if kind=="b":
            pc = tgt if ptype=="Tb" else pc+ln
            if pc is None: reports.append("  unknown branch target"); return matched,reports
        elif kind=="d":
            pc = tgt
            if pc is None: reports.append("  unknown jump target"); return matched,reports
        else:
            pc = pc ^ (t<<1)
        matched += 1
    return matched, reports

if __name__ == "__main__":
    dis, tr, spc = sys.argv[1], sys.argv[2], int(sys.argv[3],16)
    for skip in range(0, int(sys.argv[4]) if len(sys.argv)>4 else 1):
        m, rep = run(dis, tr, spc, skip)
        print(f"skip={skip}: matched {m} packets")
        for r in rep[:3]: print(r)
