// Object Tracker - Robust implementation for marker and marker-less tracking

class ObjectTracker {
    constructor() {
        this.trackingMode = 'marker';
        this.template = null;
        this.templateGray = null;
        this.lastPosition = null;
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
        if (this.templateGray) {
            try { this.templateGray.delete(); } catch(e) {}
        }
        this.template = null;
        this.templateGray = null;
        this.lastPosition = null;
        this.sam2Data = null;
        this.sam2Masks = [];
        this.sam2Centroids = [];
    }
    
    // ========== MARKER-BASED TRACKING ==========
    trackMarker(src, dst) {
        let found = false;
        let count = 0;
        
        try {
            const gray = new cv.Mat();
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            
            const thresh = new cv.Mat();
            cv.threshold(gray, thresh, 100, 255, cv.THRESH_BINARY);
            
            const contours = new cv.MatVector();
            const hierarchy = new cv.Mat();
            cv.findContours(thresh, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
            
            const frameArea = src.rows * src.cols;
            const minArea = frameArea * 0.005;
            const maxArea = frameArea * 0.3;
            
            for (let i = 0; i < contours.size(); i++) {
                const contour = contours.get(i);
                const area = cv.contourArea(contour);
                
                if (area < minArea || area > maxArea) continue;
                
                const epsilon = 0.04 * cv.arcLength(contour, true);
                const approx = new cv.Mat();
                cv.approxPolyDP(contour, approx, epsilon, true);
                
                if (approx.rows === 4 && cv.isContourConvex(approx)) {
                    const rect = cv.boundingRect(approx);
                    const aspectRatio = rect.width / rect.height;
                    
                    if (aspectRatio > 0.8 && aspectRatio < 1.2) {
                        const boundingArea = rect.width * rect.height;
                        const fillRatio = area / boundingArea;
                        
                        if (fillRatio > 0.75) {
                            this.drawDetection(dst, rect, [0, 255, 0, 255]);
                            found = true;
                            count++;
                        }
                    }
                }
                approx.delete();
            }
            
            gray.delete();
            thresh.delete();
            contours.delete();
            hierarchy.delete();
            
        } catch (e) {
            console.error('Marker tracking error:', e);
        }
        
        return { found, count };
    }
    
    // ========== MARKER-LESS TRACKING (Template Matching) ==========
    setTemplate(rect, frame) {
        try {
            // Cleanup old templates
            if (this.template) {
                try { this.template.delete(); } catch(e) {}
            }
            if (this.templateGray) {
                try { this.templateGray.delete(); } catch(e) {}
            }
            this.template = null;
            this.templateGray = null;
            
            // Validate coordinates
            let x = Math.round(rect.x);
            let y = Math.round(rect.y);
            let w = Math.round(rect.width);
            let h = Math.round(rect.height);
            
            // Clamp to frame bounds
            x = Math.max(0, x);
            y = Math.max(0, y);
            if (x + w > frame.cols) w = frame.cols - x;
            if (y + h > frame.rows) h = frame.rows - y;
            
            console.log(`Setting template: x=${x}, y=${y}, w=${w}, h=${h}, frame=${frame.cols}x${frame.rows}`);
            
            if (w < 30 || h < 30) {
                console.warn('Template too small');
                return false;
            }
            
            // Extract ROI
            const roiRect = new cv.Rect(x, y, w, h);
            const roi = frame.roi(roiRect);
            this.template = roi.clone();
            roi.delete();
            
            // Pre-compute grayscale template
            this.templateGray = new cv.Mat();
            cv.cvtColor(this.template, this.templateGray, cv.COLOR_RGBA2GRAY);
            
            // Store initial position
            this.lastPosition = { x, y, width: w, height: h };
            
            console.log(`Template captured: ${this.template.cols}x${this.template.rows}`);
            return true;
            
        } catch (e) {
            console.error('setTemplate error:', e);
            this.template = null;
            this.templateGray = null;
            return false;
        }
    }
    
    trackMarkerless(src, dst) {
        if (!this.template || !this.templateGray) {
            return false;
        }
        
        try {
            // Convert source to grayscale
            const srcGray = new cv.Mat();
            cv.cvtColor(src, srcGray, cv.COLOR_RGBA2GRAY);
            
            // Check sizes
            if (this.templateGray.rows > srcGray.rows || this.templateGray.cols > srcGray.cols) {
                srcGray.delete();
                return false;
            }
            
            // Template matching
            const resultCols = srcGray.cols - this.templateGray.cols + 1;
            const resultRows = srcGray.rows - this.templateGray.rows + 1;
            const result = new cv.Mat(resultRows, resultCols, cv.CV_32FC1);
            
            cv.matchTemplate(srcGray, this.templateGray, result, cv.TM_CCOEFF_NORMED);
            
            // Find best match
            const minMax = cv.minMaxLoc(result);
            const maxVal = minMax.maxVal;
            const maxLoc = minMax.maxLoc;
            
            srcGray.delete();
            result.delete();
            
            // Threshold for good match
            if (maxVal > 0.4) {
                const x = maxLoc.x;
                const y = maxLoc.y;
                const w = this.template.cols;
                const h = this.template.rows;
                
                // Update last known position
                this.lastPosition = { x, y, width: w, height: h };
                
                // Draw detection with confidence
                this.drawDetection(dst, { x, y, width: w, height: h }, [0, 255, 0, 255], maxVal);
                
                return true;
            }
            
            // If match failed but we have last position, show it in red
            if (this.lastPosition) {
                this.drawDetection(dst, this.lastPosition, [255, 100, 100, 255], 0);
            }
            
            return false;
            
        } catch (e) {
            console.error('trackMarkerless error:', e);
            return false;
        }
    }
    
    // ========== DRAWING HELPER ==========
    drawDetection(dst, rect, color, confidence = null) {
        const x = rect.x;
        const y = rect.y;
        const w = rect.width;
        const h = rect.height;
        
        // Bounding box
        cv.rectangle(dst, {x: x, y: y}, {x: x + w, y: y + h}, color, 3);
        
        // Corner circles
        cv.circle(dst, {x: x, y: y}, 5, [255, 255, 0, 255], -1);
        cv.circle(dst, {x: x + w, y: y}, 5, [255, 255, 0, 255], -1);
        cv.circle(dst, {x: x + w, y: y + h}, 5, [255, 255, 0, 255], -1);
        cv.circle(dst, {x: x, y: y + h}, 5, [255, 255, 0, 255], -1);
        
        // Center point
        const cx = Math.round(x + w / 2);
        const cy = Math.round(y + h / 2);
        cv.circle(dst, {x: cx, y: cy}, 6, [255, 0, 255, 255], -1);
        
        // Crosshair
        cv.line(dst, {x: cx - 12, y: cy}, {x: cx + 12, y: cy}, [255, 0, 255, 255], 2);
        cv.line(dst, {x: cx, y: cy - 12}, {x: cx, y: cy + 12}, [255, 0, 255, 255], 2);
        
        // Confidence label
        if (confidence !== null && confidence > 0) {
            const label = `${Math.round(confidence * 100)}%`;
            cv.putText(dst, label, {x: x, y: y - 8}, cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
    }
    
    // ========== SAM2 TRACKING ==========
    async loadSAM2Data(npzData) {
        try {
            this.sam2Data = npzData;
            this.sam2Masks = [];
            this.sam2Centroids = [];
            
            const parser = new NPZParser();
            const npz = await parser.parseNPZ(npzData);
            
            if (npz.masks) {
                const maskArray = npz.masks;
                if (maskArray.shape.length === 3) {
                    this.sam2Masks = parser.numpyToMat(maskArray);
                } else if (maskArray.shape.length === 2) {
                    this.sam2Masks = [parser.numpyToMat(maskArray)];
                }
                this.sam2Centroids = this.sam2Masks.map(m => this.computeCentroid(m)).filter(c => c);
            }
        } catch (e) {
            console.error('SAM2 load error:', e);
        }
    }
    
    computeCentroid(mask) {
        try {
            const moments = cv.moments(mask, false);
            if (moments.m00 === 0) return null;
            return { x: moments.m10 / moments.m00, y: moments.m01 / moments.m00 };
        } catch (e) {
            return null;
        }
    }
    
    trackSAM2(src, dst) {
        if (!this.sam2Masks || this.sam2Masks.length === 0) {
            const cx = src.cols / 2;
            const cy = src.rows / 2;
            cv.putText(dst, "Load NPZ file", {x: cx - 60, y: cy}, cv.FONT_HERSHEY_SIMPLEX, 0.7, [255, 0, 255, 255], 2);
            return { found: false, count: 0 };
        }
        
        let count = 0;
        for (let i = 0; i < this.sam2Masks.length; i++) {
            const mask = this.sam2Masks[i];
            if (!mask || mask.empty()) continue;
            
            let scaledMask = mask;
            if (mask.rows !== src.rows || mask.cols !== src.cols) {
                scaledMask = new cv.Mat();
                cv.resize(mask, scaledMask, new cv.Size(src.cols, src.rows));
            }
            
            const rect = cv.boundingRect(scaledMask);
            this.drawDetection(dst, rect, [255, 0, 255, 255]);
            
            if (scaledMask !== mask) scaledMask.delete();
            count++;
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
                    const mr = this.trackMarker(src, dst);
                    tracked = mr.found;
                    objectCount = mr.count;
                    break;
                    
                case 'markerless':
                    tracked = this.trackMarkerless(src, dst);
                    objectCount = tracked ? 1 : 0;
                    break;
                    
                case 'sam2':
                    const sr = this.trackSAM2(src, dst);
                    tracked = sr.found;
                    objectCount = sr.count;
                    break;
            }
        } catch (e) {
            console.error('processFrame error:', e);
        }
        
        return { tracked, objectCount };
    }
}
