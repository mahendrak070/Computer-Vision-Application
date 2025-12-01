// NPZ File Parser for JavaScript
// NPZ files are ZIP archives containing .npy (NumPy array) files

class NPZParser {
    constructor() {
        this.data = null;
    }
    
    /**
     * Parse NPZ file from ArrayBuffer
     * @param {ArrayBuffer} arrayBuffer - Raw NPZ file data
     * @returns {Promise<Object>} Parsed data with arrays
     */
    async parseNPZ(arrayBuffer) {
        try {
            // Load ZIP archive
            const zip = await JSZip.loadAsync(arrayBuffer);
            const result = {};
            
            // Parse each .npy file in the archive
            for (const filename in zip.files) {
                if (filename.endsWith('.npy')) {
                    const file = zip.files[filename];
                    const arrayBuffer = await file.async('arraybuffer');
                    const arrayName = filename.replace('.npy', '');
                    result[arrayName] = this.parseNPY(arrayBuffer);
                }
            }
            
            return result;
        } catch (error) {
            console.error('Error parsing NPZ file:', error);
            throw error;
        }
    }
    
    /**
     * Parse .npy (NumPy array) file
     * @param {ArrayBuffer} arrayBuffer - Raw .npy file data
     * @returns {Object} Parsed array with data and shape
     */
    parseNPY(arrayBuffer) {
        const view = new DataView(arrayBuffer);
        const uint8View = new Uint8Array(arrayBuffer);
        let offset = 0;
        
        // Check magic number (first 6 bytes: 0x93 'N' 'U' 'M' 'P' 'Y')
        // Magic should be: 0x93 0x4E 0x55 0x4D 0x50 0x59
        const expectedMagic = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59];
        let isValidMagic = true;
        
        for (let i = 0; i < 6; i++) {
            if (uint8View[i] !== expectedMagic[i]) {
                isValidMagic = false;
                break;
            }
        }
        
        if (!isValidMagic) {
            // Log for debugging
            const actualMagic = Array.from(uint8View.slice(0, 6));
            console.log('Expected magic:', expectedMagic.map(b => '0x' + b.toString(16)).join(' '));
            console.log('Actual magic:', actualMagic.map(b => '0x' + b.toString(16)).join(' '));
            
            // Check if it's at least close (maybe version difference)
            if (uint8View[0] === 0x93 && uint8View[1] === 0x4E) {
                console.warn('Magic number partially matches, continuing...');
            } else {
                throw new Error('Invalid .npy file format - magic number mismatch. Expected: 0x93 0x4E 0x55 0x4D 0x50 0x59');
            }
        }
        offset = 6;
        
        // Read version
        const version = view.getUint8(offset);
        offset += 1;
        
        console.log('NPY version:', version);
        console.log('File size:', arrayBuffer.byteLength, 'bytes');
        console.log('Bytes at offset 6-10:', Array.from(uint8View.slice(6, 12)).map(b => '0x' + b.toString(16)).join(' '));
        
        // Read header length
        let headerLength;
        let headerStartOffset = offset + 2; // Default: header starts after version + length bytes
        let headerContentStart = -1;
        
        if (version === 1) {
            // Version 1: header length is 2 bytes little-endian
            const byte0 = uint8View[offset];
            const byte1 = uint8View[offset + 1];
            headerLength = byte0 | (byte1 << 8); // Little-endian
            console.log(`Header length bytes: 0x${byte0.toString(16)} 0x${byte1.toString(16)} = ${headerLength}`);
            
            // Check if header length seems wrong (too large)
            // If so, try to find the actual header start
            if (headerLength > 1000 || headerLength < 10) {
                console.warn(`Suspicious header length: ${headerLength}. Searching for actual header...`);
                
                // Search for '{' which should start the header (usually right after header length)
                const searchStart = offset + 2;
                const searchEnd = Math.min(searchStart + 100, arrayBuffer.byteLength);
                
                for (let i = searchStart; i < searchEnd; i++) {
                    if (uint8View[i] === 0x7B) { // '{' character
                        headerContentStart = i;
                        break;
                    }
                }
                
                if (headerContentStart > 0) {
                    console.log(`Found header start at offset ${headerContentStart}`);
                    // Find the end of the header dict
                    let headerEnd = searchEnd;
                    let braceCount = 0;
                    
                    for (let i = headerContentStart; i < Math.min(headerContentStart + 500, arrayBuffer.byteLength); i++) {
                        const byte = uint8View[i];
                        if (byte === 0x7B) { // '{'
                            braceCount++;
                        } else if (byte === 0x7D) { // '}'
                            braceCount--;
                            if (braceCount === 0) {
                                headerEnd = i + 1;
                                break;
                            }
                        }
                    }
                    
                    // Calculate actual header length from headerStartOffset to end of dict (including padding)
                    const actualHeaderContentLength = headerEnd - headerContentStart;
                    // Pad to 16-byte boundary for version 1
                    const padding = (16 - (actualHeaderContentLength % 16)) % 16;
                    headerLength = actualHeaderContentLength + padding;
                    headerStartOffset = headerContentStart; // Start reading from where header actually begins
                    console.log(`Recalculated header length: ${headerLength} (content: ${actualHeaderContentLength}, padding: ${padding})`);
                } else {
                    // Try reading as if bytes are swapped
                    const swappedLength = byte1 | (byte0 << 8);
                    if (swappedLength < 1000 && swappedLength > 10) {
                        console.log(`Trying swapped byte order: ${swappedLength}`);
                        headerLength = swappedLength;
                    } else {
                        throw new Error(`Invalid header length: ${headerLength}. Could not find header start.`);
                    }
                }
            }
            offset += 2;
        } else if (version === 2) {
            headerLength = view.getUint32(offset, true); // Little-endian
            offset += 4;
        } else {
            throw new Error(`Unsupported NPY version: ${version}`);
        }
        
        console.log('Final header length:', headerLength);
        
        // Use the correct starting offset for reading the header
        const headerReadOffset = headerContentStart > 0 ? headerContentStart : offset;
        
        // Ensure we don't read past the end of the file
        if (headerReadOffset + headerLength > arrayBuffer.byteLength) {
            // Try to read what we can
            const availableBytes = arrayBuffer.byteLength - headerReadOffset;
            console.warn(`Header length (${headerLength}) exceeds available bytes (${availableBytes}). Reading what we can.`);
            headerLength = availableBytes;
        }
        
        // Read header (Python dict as string)
        // Note: Header is padded to align to 16-byte boundary in version 1, 64-byte in version 2
        const headerBytes = uint8View.slice(headerReadOffset, headerReadOffset + headerLength);
        
        // Convert to string, but stop at first null byte or end of meaningful content
        let headerStr = '';
        for (let i = 0; i < headerBytes.length; i++) {
            const byte = headerBytes[i];
            if (byte === 0) {
                break; // Stop at null byte
            }
            const char = String.fromCharCode(byte);
            headerStr += char;
        }
        
        // Trim trailing whitespace
        headerStr = headerStr.trim();
        
        // Find actual header end (before padding spaces)
        let actualHeaderEnd = headerStr.length;
        for (let i = headerStr.length - 1; i >= 0; i--) {
            const char = headerStr[i];
            if (char !== ' ' && char !== '\n' && char !== '\r' && char !== '\t') {
                actualHeaderEnd = i + 1;
                break;
            }
        }
        const actualHeaderStr = headerStr.substring(0, actualHeaderEnd);
        
        // Move offset past header (including padding)
        // If we found the header start manually, calculate from there
        if (headerContentStart > 0) {
            offset = headerContentStart + headerLength;
        } else {
            offset += headerLength;
        }
        
        console.log('Header (trimmed, length:', actualHeaderStr.length, '):', actualHeaderStr.substring(0, 200));
        
        // Parse header
        const header = this.parseHeader(actualHeaderStr);
        
        // Read data
        const dataLength = arrayBuffer.byteLength - offset;
        
        if (dataLength < 0) {
            throw new Error(`Invalid data length: ${dataLength}. Header length may be incorrect.`);
        }
        
        if (dataLength === 0) {
            throw new Error('No data found after header. File may be corrupted.');
        }
        
        const data = new Uint8Array(arrayBuffer, offset, dataLength);
        
        // Convert to typed array based on dtype
        const typedArray = this.convertToTypedArray(data, header.descr, header.shape);
        
        return {
            data: typedArray,
            shape: header.shape,
            dtype: header.descr,
            fortran_order: header.fortran_order
        };
    }
    
    /**
     * Parse header string (Python dict format)
     */
    parseHeader(headerStr) {
        const header = {};
        
        // Extract shape - handle different formats
        // Try with parentheses first: shape: (2, 480, 640) or shape:(2,480,640)
        let shapeMatch = headerStr.match(/shape\s*:\s*\(([^)]+)\)/);
        
        if (!shapeMatch) {
            // Try without parentheses: shape: 2, 480, 640
            shapeMatch = headerStr.match(/shape\s*:\s*([^\s,}]+(?:,\s*[^\s,}]+)*)/);
        }
        
        if (shapeMatch) {
            const shapeStr = shapeMatch[1].trim();
            if (shapeStr && shapeStr !== '') {
                // Split by comma and parse each dimension
                header.shape = shapeStr.split(',').map(s => {
                    const trimmed = s.trim();
                    const parsed = parseInt(trimmed);
                    if (isNaN(parsed)) {
                        console.warn('Could not parse shape dimension:', trimmed);
                        return null;
                    }
                    return parsed;
                }).filter(n => n !== null);
            } else {
                header.shape = [];
            }
        } else {
            // Last resort: try to find shape anywhere in the string
            const shapePattern = /\((\d+(?:\s*,\s*\d+)*)\)/;
            const fallbackMatch = headerStr.match(shapePattern);
            if (fallbackMatch) {
                const shapeStr = fallbackMatch[1];
                header.shape = shapeStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
            } else {
                console.warn('Could not parse shape from header:', headerStr.substring(0, 200));
                header.shape = [];
            }
        }
        
        // Extract dtype - handle different quote styles and formats
        // Examples: 'descr': '|u1', "descr": "<f4", descr: '|u1'
        // The header format is: 'descr': '|u1'
        let dtypeMatch = null;
        
        // Pattern 1: 'descr': '|u1' (with single quotes - most common in NumPy)
        // Match: 'descr' followed by : followed by 'value'
        dtypeMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
        
        // Pattern 2: "descr": "|u1" (with double quotes)
        if (!dtypeMatch) {
            dtypeMatch = headerStr.match(/"descr"\s*:\s*"([^"]+)"/);
        }
        
        // Pattern 3: descr: '|u1' (key without quotes, value with quotes)
        if (!dtypeMatch) {
            dtypeMatch = headerStr.match(/descr\s*:\s*['"]([^'"]+)['"]/);
        }
        
        // Pattern 4: descr: |u1 (without quotes)
        if (!dtypeMatch) {
            dtypeMatch = headerStr.match(/descr\s*:\s*([^\s,}]+)/);
        }
        
        // Pattern 5: Last resort - find any quoted string after 'descr'
        if (!dtypeMatch) {
            const descrIndex = headerStr.indexOf("'descr'");
            if (descrIndex >= 0) {
                const afterDescr = headerStr.substring(descrIndex + 7);
                dtypeMatch = afterDescr.match(/\s*:\s*'([^']+)'/);
            }
        }
        
        // Pattern 6: Even more flexible - find descr anywhere and get next quoted value
        if (!dtypeMatch) {
            const descrIndex = headerStr.indexOf('descr');
            if (descrIndex >= 0) {
                const afterDescr = headerStr.substring(descrIndex);
                // Look for colon followed by quoted string
                dtypeMatch = afterDescr.match(/:\s*['"]([^'"]+)['"]/);
            }
        }
        
        if (dtypeMatch) {
            header.descr = dtypeMatch[1].trim();
            console.log('Extracted dtype:', header.descr);
            // Handle native byte order: |u1 -> <u1 (assume little-endian)
            if (header.descr[0] === '|') {
                header.descr = '<' + header.descr.slice(1);
                console.log('Converted native byte order to little-endian:', header.descr);
            }
        } else {
            console.warn('Could not parse dtype from header:', headerStr.substring(0, 100));
            header.descr = '<f4'; // Default to float32
        }
        
        // Extract fortran_order
        const fortranMatch = headerStr.match(/fortran_order\s*:\s*(True|False)/);
        header.fortran_order = fortranMatch && fortranMatch[1] === 'True';
        
        console.log('Parsed header - shape:', header.shape, 'dtype:', header.descr);
        
        return header;
    }
    
    /**
     * Convert raw data to typed array based on dtype
     */
    convertToTypedArray(data, dtype, shape) {
        const totalElements = shape.reduce((a, b) => a * b, 1);
        
        // Parse dtype (e.g., '<f4' means little-endian float32, 'u1' means uint8)
        let endian = 'little';
        let typeChar = dtype[0];
        let bytes = 1;
        
        if (dtype.length > 1 && (dtype[0] === '<' || dtype[0] === '>')) {
            // Endianness specified
            endian = dtype[0] === '<' ? 'little' : 'big';
            typeChar = dtype[1];
            bytes = parseInt(dtype.slice(2)) || 1;
        } else if (dtype.length > 1 && dtype[0] === 'u') {
            // uint type without endianness
            typeChar = 'u';
            bytes = parseInt(dtype.slice(1)) || 1;
        } else if (dtype.length > 1 && dtype[0] === 'i') {
            // int type without endianness
            typeChar = 'i';
            bytes = parseInt(dtype.slice(1)) || 1;
        } else if (dtype.length > 1 && dtype[0] === 'f') {
            // float type without endianness
            typeChar = 'f';
            bytes = parseInt(dtype.slice(1)) || 4;
        }
        
        let TypedArray;
        switch (typeChar) {
            case 'f': // float
                TypedArray = bytes === 4 ? Float32Array : Float64Array;
                break;
            case 'i': // int
                TypedArray = bytes === 1 ? Int8Array : 
                            bytes === 2 ? Int16Array : 
                            bytes === 4 ? Int32Array : Int32Array;
                break;
            case 'u': // uint
                TypedArray = bytes === 1 ? Uint8Array : 
                            bytes === 2 ? Uint16Array : 
                            bytes === 4 ? Uint32Array : Uint32Array;
                break;
            default:
                console.warn('Unknown dtype:', dtype, '- defaulting to Uint8Array');
                TypedArray = Uint8Array;
                bytes = 1;
        }
        
        // Create typed array view
        const expectedBytes = totalElements * bytes;
        const availableBytes = data.byteLength;
        
        console.log(`Converting array: shape=${shape}, dtype=${dtype}, bytes per element=${bytes}, total elements=${totalElements}, expected bytes=${expectedBytes}, available=${availableBytes}`);
        
        if (expectedBytes > availableBytes) {
            console.warn(`Expected ${expectedBytes} bytes but got ${availableBytes}. Using available bytes.`);
        }
        
        // Use the minimum of expected and available bytes
        const bytesToRead = Math.min(expectedBytes, availableBytes);
        
        // Ensure we read a multiple of the element size for proper alignment
        const alignedBytes = Math.floor(bytesToRead / bytes) * bytes;
        
        if (alignedBytes === 0) {
            throw new Error(`No data available. Expected ${expectedBytes} bytes but got ${availableBytes}`);
        }
        
        const buffer = data.buffer.slice(data.byteOffset, data.byteOffset + alignedBytes);
        const typedArray = new TypedArray(buffer);
        
        console.log(`Created ${TypedArray.name} with ${typedArray.length} elements`);
        
        return typedArray;
    }
    
    /**
     * Convert parsed numpy array to OpenCV Mat
     * @param {Object} npArray - Parsed numpy array
     * @returns {cv.Mat} OpenCV Mat object
     */
    numpyToMat(npArray) {
        const shape = npArray.shape;
        
        if (shape.length === 2) {
            // 2D array: single mask
            const rows = shape[0];
            const cols = shape[1];
            const mat = new cv.Mat(rows, cols, cv.CV_8UC1);
            
            const data = npArray.data;
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    const idx = i * cols + j;
                    mat.data[idx] = data[idx] > 0 ? 255 : 0;
                }
            }
            
            return mat;
        } else if (shape.length === 3) {
            // 3D array: multiple masks (N, H, W)
            // Return array of Mats
            const numMasks = shape[0];
            const rows = shape[1];
            const cols = shape[2];
            const masks = [];
            
            const data = npArray.data;
            for (let n = 0; n < numMasks; n++) {
                const mat = new cv.Mat(rows, cols, cv.CV_8UC1);
                for (let i = 0; i < rows; i++) {
                    for (let j = 0; j < cols; j++) {
                        const idx = n * rows * cols + i * cols + j;
                        mat.data[i * cols + j] = data[idx] > 0 ? 255 : 0;
                    }
                }
                masks.push(mat);
            }
            
            return masks;
        }
        
        throw new Error('Unsupported array shape');
    }
}

