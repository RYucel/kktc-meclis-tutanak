"""Pure-Python MS-CFB (OLE2) + MS-DOC (Word 6/95 and Word 97-2003) text extractor fallback.

Recovers text from:
- Word 6.0/95 .doc files blocked by Microsoft Office Trust Center File Block settings
- Word 97-2003 .doc files with corrupted tables that Word COM refuses to open
- Extension-mismatched OLE2 .doc files saved with .docx extension
"""

import struct
from pathlib import Path

ENDOFCHAIN = 0xFFFFFFFE
FREESECT = 0xFFFFFFFF
FATSECT = 0xFFFFFFFD
DIFSECT = 0xFFFFFFFC


class CFBReader:
    """Minimal spec-compliant Microsoft Compound File Binary (OLE2) reader."""

    def __init__(self, data: bytes):
        self.data = data
        if data[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            raise ValueError("Not an OLE2 CFB file")
        sector_shift = struct.unpack_from("<H", data, 0x1E)[0]
        mini_sector_shift = struct.unpack_from("<H", data, 0x20)[0]
        self.sector_size = 1 << sector_shift
        self.mini_sector_size = 1 << mini_sector_shift
        self.dir_first_sector = struct.unpack_from("<I", data, 0x30)[0]
        self.mini_cutoff = struct.unpack_from("<I", data, 0x38)[0]
        self.minifat_first_sector = struct.unpack_from("<I", data, 0x3C)[0]
        self.difat_first_sector = struct.unpack_from("<I", data, 0x44)[0]
        self.difat_count = struct.unpack_from("<I", data, 0x48)[0]

        # Build DIFAT -> FAT sector list
        fat_sectors = list(struct.unpack_from("<109I", data, 0x4C))
        dif_sec = self.difat_first_sector
        for _ in range(self.difat_count):
            if dif_sec in (ENDOFCHAIN, FREESECT):
                break
            buf = self._read_sector(dif_sec)
            n_entries = (self.sector_size // 4) - 1
            fat_sectors.extend(struct.unpack_from(f"<{n_entries}I", buf, 0))
            dif_sec = struct.unpack_from("<I", buf, n_entries * 4)[0]

        # Build FAT
        self.fat: list[int] = []
        per_sec = self.sector_size // 4
        for sec in fat_sectors:
            if sec >= 0xFFFFFFFA:
                continue
            buf = self._read_sector(sec)
            if len(buf) == self.sector_size:
                self.fat.extend(struct.unpack_from(f"<{per_sec}I", buf, 0))

        # Build MiniFAT
        self.minifat: list[int] = []
        if self.minifat_first_sector not in (ENDOFCHAIN, FREESECT):
            mf_bytes = self._read_chain(self.minifat_first_sector)
            count = len(mf_bytes) // 4
            self.minifat = list(struct.unpack_from(f"<{count}I", mf_bytes, 0))

        # Parse Directory entries
        dir_bytes = self._read_chain(self.dir_first_sector)
        self.entries: dict[str, tuple[int, int, int]] = {}
        root_start = ENDOFCHAIN
        root_size = 0
        for off in range(0, len(dir_bytes) - 127, 128):
            name_len = struct.unpack_from("<H", dir_bytes, off + 0x40)[0]
            obj_type = dir_bytes[off + 0x42]
            if obj_type == 0 or name_len < 2:
                continue
            name = dir_bytes[off : off + min(64, name_len - 2)].decode("utf-16le", errors="ignore")
            start_sec = struct.unpack_from("<I", dir_bytes, off + 0x74)[0]
            size = struct.unpack_from("<Q", dir_bytes, off + 0x78)[0]
            if self.sector_size == 512:
                size &= 0xFFFFFFFF
            if obj_type == 5:  # Root Entry
                root_start = start_sec
                root_size = size
            elif obj_type == 2:  # Stream
                self.entries[name] = (start_sec, size, obj_type)

        self.ministream = (
            self._read_chain(root_start, root_size)
            if root_start not in (ENDOFCHAIN, FREESECT)
            else b""
        )

    def _read_sector(self, sec: int) -> bytes:
        off = (sec + 1) * self.sector_size
        return self.data[off : off + self.sector_size]

    def _read_chain(self, start: int, max_len: int | None = None) -> bytes:
        chunks = []
        sec = start
        seen = set()
        total = 0
        while sec not in (ENDOFCHAIN, FREESECT) and sec < len(self.fat) and sec not in seen:
            seen.add(sec)
            b = self._read_sector(sec)
            chunks.append(b)
            total += len(b)
            if max_len is not None and total >= max_len:
                break
            sec = self.fat[sec]
        res = b"".join(chunks)
        return res[:max_len] if max_len is not None else res

    def read_stream(self, name: str) -> bytes | None:
        if name not in self.entries:
            return None
        start, size, _ = self.entries[name]
        if size < self.mini_cutoff and self.ministream and self.minifat:
            chunks = []
            sec = start
            seen = set()
            total = 0
            while sec not in (ENDOFCHAIN, FREESECT) and sec < len(self.minifat) and sec not in seen:
                seen.add(sec)
                off = sec * self.mini_sector_size
                chunks.append(self.ministream[off : off + self.mini_sector_size])
                total += self.mini_sector_size
                if total >= size:
                    break
                sec = self.minifat[sec]
            return b"".join(chunks)[:size]
        return self._read_chain(start, size)


def extract_ole_doc_text(path: Path) -> str:
    data = path.read_bytes()
    cfb = CFBReader(data)
    wd = cfb.read_stream("WordDocument")
    if not wd or len(wd) < 0x1B0:
        raise ValueError("Missing or truncated WordDocument stream")

    wIdent, nFib = struct.unpack_from("<HH", wd, 0)
    flags = struct.unpack_from("<H", wd, 0x0A)[0]

    # Case 1: Word 6.0 / 95 (nFib < 0x00C0 or wIdent == 0xA5DC)
    if wIdent == 0xA5DC or nFib < 0x00C0:
        fcMin, fcMac = struct.unpack_from("<II", wd, 0x18)
        if fcMin < fcMac <= len(wd):
            # Word 95 Turkish documents use cp1254
            return wd[fcMin:fcMac].decode("cp1254", errors="replace")

    # Case 2: Word 97-2003 (MS-DOC Piece Table via 1Table or 0Table)
    table_name = "1Table" if (flags & 0x0200) else "0Table"
    tbl = cfb.read_stream(table_name) or cfb.read_stream("1Table") or cfb.read_stream("0Table")
    if tbl and len(wd) >= 0x1AA:
        ccpText = struct.unpack_from("<I", wd, 0x4C)[0]
        fcClx, lcbClx = struct.unpack_from("<II", wd, 0x1A2)
        if lcbClx > 0 and fcClx + lcbClx <= len(tbl):
            clx = tbl[fcClx : fcClx + lcbClx]
            pos = 0
            # Skip Prc entries (clxt == 0x01)
            while pos < len(clx) and clx[pos] == 0x01:
                if pos + 3 > len(clx):
                    break
                cb = struct.unpack_from("<H", clx, pos + 1)[0]
                pos += 3 + cb
            # Parse Pcdt (clxt == 0x02)
            if pos + 5 <= len(clx) and clx[pos] == 0x02:
                lcb = struct.unpack_from("<I", clx, pos + 1)[0]
                plc = clx[pos + 5 : pos + 5 + lcb]
                # PlcPcd: (n+1) CPs (4 bytes each) + n PCDs (8 bytes each) => lcb = 4*(n+1) + 8*n = 12*n + 4
                n = (len(plc) - 4) // 12
                if n > 0:
                    cps = struct.unpack_from(f"<{n + 1}I", plc, 0)
                    pcd_base = 4 * (n + 1)
                    pieces = []
                    for i in range(n):
                        cp_start, cp_end = cps[i], cps[i + 1]
                        if ccpText and cp_start >= ccpText:
                            break
                        if ccpText and cp_end > ccpText:
                            cp_end = ccpText
                        char_len = cp_end - cp_start
                        if char_len <= 0:
                            continue
                        fc_raw = struct.unpack_from("<I", plc, pcd_base + i * 8 + 2)[0]
                        is_compressed = bool(fc_raw & 0x40000000)
                        fc = fc_raw & 0x3FFFFFFF
                        if is_compressed:
                            byte_off = fc // 2
                            chunk = wd[byte_off : byte_off + char_len]
                            pieces.append(chunk.decode("cp1254", errors="replace"))
                        else:
                            byte_off = fc
                            chunk = wd[byte_off : byte_off + char_len * 2]
                            pieces.append(chunk.decode("utf-16le", errors="replace"))
                    text = "".join(pieces)
                    if len(text.strip()) > 20:
                        return text

    # Fallback: fcMin..fcMac if present
    fcMin, fcMac = struct.unpack_from("<II", wd, 0x18)
    if fcMin < fcMac <= len(wd):
        return wd[fcMin:fcMac].decode("cp1254", errors="replace")

    raise ValueError("Could not decode WordDocument text")
