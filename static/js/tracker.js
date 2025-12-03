// Object Tracker - Robust implementation for marker and marker-less tracking

class ObjectTracker {
    constructor() {
        this.trackingMode = 'marker';
        this.isTracking = false;
        this.template = null;
        this.templateRect = null;
        this.sam2Data = null;
        this.sam2Masks = [];
        this.sam2Centroids = [];
        
        console.log('ObjectTracker initialized');
    }
    
    setMode(mode) {
        this.trackingMode = mode;
        this.reset();
        console.log('Tracking mode set to:', mode);
    }
    
    reset() {
        if (this.template) {
            try { this.template.delete(); } catch(e) {}
        }
        this.template = null;
        this.templateRect = null;
        this.sam2Data = null;
        this.sam2Masks = [];
        this.sam2Centroids = [];
    }
    
    // ========== MARKER-BASED TRACKING (ArUco-like squares) ==========
    trackMarker(src, dst) {
        let found = false;
        let count = 0;
        
        try {
            // Convert to grayscale
            const gray = new cv.Mat();
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            
            // Apply binary threshold - look for high contrast markers
            const thresh = new cv.Mat();
            cv.threshold(gray, thresh, 100, 255, cv.THRESH_BINARY);
            
            // Find contours
            const contours = new cv.MatVector();
            const hierarchy = new cv.Mat();
            cv.findContours(thresh, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
            
            // Frame dimensions for filtering
            const frameArea = src.rows * src.cols;
            const minArea = frameArea * 0.005;  // Min 0.5% of frame
            const maxArea = frameArea * 0.3;    // Max 30% of frame
            
            for (let i = 0; i < contours.size(); i++) {
                const contour = contours.get(i);
                const area = cv.contourArea(contour);
                
                // Filter by area
                if (area < minArea || area > maxArea) continue;
                
                // Approximate to polygon
                const epsilon = 0.04 * cv.arcLength(contour, true);
                const approx = new cv.Mat();
                cv.approxPolyDP(contour, approx, epsilon, true);
                
                // Must be exactly 4 corners (quadrilateral)
                if (approx.rows === 4) {
                    // Check if convex
                    if (cv.isContourConvex(approx)) {
                        const rect = cv.boundingRect(approx);
                        const aspectRatio = rect.width / rect.height;
                        
                        // Must be roughly square (aspect ratio 0.8 to 1.2)
                        if (aspectRatio > 0.8 && aspectRatio < 1.2) {
                            // Check fill ratio (area should match bounding rect closely)
                            const boundingArea = rect.width * rect.height;
                            const fillRatio = area / boundingArea;
                            
                            // Good markers have fill ratio > 0.8
                            if (fillRatio > 0.75) {
                                this.drawMarkerDetection(dst, rect);
                                found = true;
                                count++;
                            }
                        }
                    }
                }
                approx.delete();
            }
            
            // Cleanup
            gray.delete();
            thresh.delete();
            contours.delete();
            hierarchy.delete();
            
        } catch (e) {
            console.error('Marker tracking error:', e);
        }
        
        return { found, count };
    }
    
    drawMarkerDetection(dst, rect) {
        // Green bounding box
        cv.rectangle(dst, {x: rect.x, y: rect.y}, {x: rect.x + rect.width, y: rect.y + rect.height}, [0, 255, 0, 255], 3);
        
        // Yellow corners
        const corners = [
            {x: rect.x, y: rect.y},
            {x: rect.x + rect.width, y: rect.y},
            {x: rect.x + rect.width, y: rect.y + rect.height},
            {x: rect.x, y: rect.y + rect.height}
        ];
        corners.forEach(c => {
            cv.circle(dst, c, 5, [255, 255, 0, 255], -1);
        });
        
        // Center point (magenta)
        const cx = Math.round(rect.x + rect.width / 2);
        const cy = Math.round(rect.y + rect.height / 2);
        cv.circle(dst, {x: cx, y: cy}, 6, [255, 0, 255, 255], -1);
    }
    
    // ========== MARKER-LESS TRACKING (Template Matching) ==========
    setTemplate(rect, frame) {
        try {
            // Cleanup old template
            if (this.template) {
                try { this.template.delete(); } catch(e) {}
                this.template = null;
            }
            
            // Validate and clamp coordinates
            const x = Math.max(0, Math.floor(rect.x));
            const y = Math.max(0, Math.floor(rect.y));
            let w = Math.floor(rect.width);
            let h = Math.floor(rect.height);
            
            // Ensure within frame bounds
            if (x + w > frame.cols) w = frame.cols - x;
            if (y + h > frame.rows) h = frame.rows - y;
            
            if (w < 20 || h < 20) {
                console.warn('Template too small:', w, 'x', h);
                return false;
            }
            
            // Extract template region
            const roiRect = new cv.Rect(x, y, w, h);
            const roi = frame.roi(roiRect);
            this.template = roi.clone();
            roi.delete();
            
            this.templateRect = { x, y, width: w, height: h };
            
            console.log(`Template captured: ${w}x${h} at (${x}, ${y})`);
            return true;
        } catch (e) {
            console.error('Error setting template:', e);
            this.template = null;
            return false;
        }
    }
    
    trackMarkerless(src, dst) {
        if (!this.template) {
            return false;
        }
        
        try {
            // Check if template is valid
            if (this.template.empty()) {
                console.warn('Template is empty');
                return false;
            }
            
            // Convert both to grayscale
            const srcGray = new cv.Mat();
            const templateGray = new cv.Mat();
            
            cv.cvtColor(src, srcGray, cv.COLOR_RGBA2GRAY);
            cv.cvtColor(this.template, templateGray, cv.COLOR_RGBA2GRAY);
            
            // Validate sizes
            if (templateGray.rows > srcGray.rows || templateGray.cols > srcGray.cols) {
                console.warn('Template larger than source');
                srcGray.delete();
                templateGray.delete();
                return false;
            }
            
            // Perform template matching
            const result = new cv.Mat();
            cv.matchTemplate(srcGray, templateGray, result, cv.TM_CCOEFF_NORMED);
            
            // Find best match
            const minMax = cv.minMaxLoc(result);
            const confidence = minMax.maxVal;
            const maxLoc = minMax.maxLoc;
            
            // Cleanup
            srcGray.delete();
            templateGray.delete();
            result.delete();
            
            // Only show if confidence > 0.5
            if (confidence > 0.5) {
                const w = this.template.cols;
                const h = this.template.rows;
                this.drawTemplateDetection(dst, maxLoc.x, maxLoc.y, w, h, confidence);
                return true;
            }
            
            return false;
        } catch (e) {
            console.error('Template matching error:', e);
            return false;
        }
    }
    
    drawTemplateDetection(dst, x, y, w, h, confidence) {
        // Green bounding box
        cv.rectangle(dst, {x: x, y: y}, {x: x + w, y: y + h}, [0, 255, 0, 255], 3);
        
        // Cyan corners
        const corners = [
            {x: x, y: y},
            {x: x + w, y: y},
            {x: x + w, y: y + h},
            {x: x, y: y + h}
        ];
        corners.forEach(c => {
            cv.circle(dst, c, 5, [0, 255, 255, 255], -1);
        });
        
        // White center
        const cx = Math.round(x + w / 2);
        const cy = Math.round(y + h / 2);
        cv.circle(dst, {x: cx, y: cy}, 6, [255, 255, 255, 255], -1);
        
        // Confidence label
        const confText = `${Math.round(confidence * 100)}%`;
        cv.putText(dst, confText, {x: x, y: y - 10}, cv.FONT_HERSHEY_SIMPLEX, 0.6, [0, 255, 0, 255], 2);
    }
    
    // ========== SAM2 SEGMENTATION TRACKING ==========
    async loadSAM2Data(npzData) {
        try {
            this.sam2Data = npzData;
            this.sam2Masks = [];
            this.sam2Centroids = [];
            
            const parser = new NPZParser();
            const npz = await parser.parseNPZ(npzData);
            
            console.log('NPZ parsed, keys:', Object.keys(npz));
            
            if (npz.masks) {
                const maskArray = npz.masks;
                if (maskArray.shape.length === 3) {
                    this.sam2Masks = parser.numpyToMat(maskArray);
                } else if (maskArray.shape.length === 2) {
                    this.sam2Masks = [parser.numpyToMat(maskArray)];
                }
                
                this.sam2Centroids = this.sam2Masks.map(mask => this.computeCentroid(mask)).filter(c => c);
                console.log(`Loaded ${this.sam2Masks.length} masks`);
            }
        } catch (e) {
            console.error('SAM2 load error:', e);
            this.sam2Masks = [];
            this.sam2Centroids = [];
        }
    }
    
    computeCentroid(mask) {
        try {
            const moments = cv.moments(mask, false);
            if (moments.m00 === 0) return null;
            return {
                x: moments.m10 / moments.m00,
                y: moments.m01 / moments.m00
            };
        } catch (e) {
            return null;
        }
    }
    
    trackSAM2(src, dst) {
        if (!this.sam2Masks || this.sam2Masks.length === 0) {
            // Show placeholder
            const x = Math.floor(src.cols * 0.3);
            const y = Math.floor(src.rows * 0.3);
            const w = Math.floor(src.cols * 0.4);
            const h = Math.floor(src.rows * 0.4);
            
            cv.rectangle(dst, {x: x, y: y}, {x: x + w, y: y + h}, [255, 0, 255, 255], 2);
            cv.putText(dst, "Load NPZ", {x: x + w/2 - 40, y: y + h/2}, cv.FONT_HERSHEY_SIMPLEX, 0.6, [255, 0, 255, 255], 2);
            
            return { found: false, count: 0 };
        }
        
        let count = 0;
        
        try {
            for (let i = 0; i < this.sam2Masks.length; i++) {
                const mask = this.sam2Masks[i];
                const centroid = this.sam2Centroids[i];
                
                if (!mask || mask.empty()) continue;
                
                let scaledMask = mask;
                if (mask.rows !== src.rows || mask.cols !== src.cols) {
                    scaledMask = new cv.Mat();
                    cv.resize(mask, scaledMask, new cv.Size(src.cols, src.rows));
                }
                
                const rect = cv.boundingRect(scaledMask);
                cv.rectangle(dst, {x: rect.x, y: rect.y}, {x: rect.x + rect.width, y: rect.y + rect.height}, [255, 0, 255, 255], 3);
                
                if (centroid) {
                    const cx = Math.round(centroid.x * (src.cols / mask.cols));
                    const cy = Math.round(centroid.y * (src.rows / mask.rows));
                    cv.circle(dst, {x: cx, y: cy}, 8, [255, 0, 255, 255], -1);
                }
                
                if (scaledMask !== mask) scaledMask.delete();
                count++;
            }
        } catch (e) {
            console.error('SAM2 tracking error:', e);
        }
        
        return { found: count > 0, count };
    }
    
    // ========== MAIN PROCESSING ==========
    processFrame(src, dst) {
        let tracked = false;
        let objectCount = 0;
        
        try {
            switch (this.trackingMode) {
                case 'marker':
                    const markerResult = this.trackMarker(src, dst);
                    tracked = markerResult.found;
                    objectCount = markerResult.count;
                    break;
                    
                case 'markerless':
                    tracked = this.trackMarkerless(src, dst);
                    objectCount = tracked ? 1 : 0;
                    break;
                    
                case 'sam2':
                    const sam2Result = this.trackSAM2(src, dst);
                    tracked = sam2Result.found;
                    objectCount = sam2Result.count;
                    break;
            }
        } catch (e) {
            console.error('processFrame error:', e);
        }
        
        return { tracked, objectCount };
    }
}
