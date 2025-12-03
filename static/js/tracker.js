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
    
    // ========== MARKER-BASED TRACKING ==========
    trackMarker(src, dst) {
        let found = false;
        let count = 0;
        
        try {
            // Convert to grayscale
            const gray = new cv.Mat();
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            
            // Apply adaptive threshold to find dark markers on light background
            const thresh = new cv.Mat();
            cv.adaptiveThreshold(gray, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2);
            
            // Find contours
            const contours = new cv.MatVector();
            const hierarchy = new cv.Mat();
            cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            
            const minArea = (src.rows * src.cols) * 0.002;  // Min 0.2% of frame
            const maxArea = (src.rows * src.cols) * 0.4;    // Max 40% of frame
            
            for (let i = 0; i < contours.size(); i++) {
                const contour = contours.get(i);
                const area = cv.contourArea(contour);
                
                if (area > minArea && area < maxArea) {
                    // Approximate to polygon
                    const epsilon = 0.02 * cv.arcLength(contour, true);
                    const approx = new cv.Mat();
                    cv.approxPolyDP(contour, approx, epsilon, true);
                    
                    // Check for square-like shape (4 corners)
                    if (approx.rows >= 4 && approx.rows <= 6) {
                        const rect = cv.boundingRect(contour);
                        const aspectRatio = rect.width / rect.height;
                        
                        // Check aspect ratio (should be roughly square)
                        if (aspectRatio > 0.6 && aspectRatio < 1.5) {
                            // Draw detection
                            this.drawMarkerDetection(dst, rect);
                            found = true;
                            count++;
                        }
                    }
                    approx.delete();
                }
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
        const pt1 = {x: rect.x, y: rect.y};
        const pt2 = {x: rect.x + rect.width, y: rect.y + rect.height};
        cv.rectangle(dst, pt1, pt2, [0, 255, 0, 255], 3);
        
        // Yellow corners
        const corners = [
            {x: rect.x, y: rect.y},
            {x: rect.x + rect.width, y: rect.y},
            {x: rect.x + rect.width, y: rect.y + rect.height},
            {x: rect.x, y: rect.y + rect.height}
        ];
        corners.forEach(c => {
            cv.circle(dst, c, 6, [255, 255, 0, 255], -1);
        });
        
        // Center with crosshair (magenta)
        const cx = Math.round(rect.x + rect.width / 2);
        const cy = Math.round(rect.y + rect.height / 2);
        cv.circle(dst, {x: cx, y: cy}, 8, [255, 0, 255, 255], -1);
        cv.line(dst, {x: cx - 15, y: cy}, {x: cx + 15, y: cy}, [255, 0, 255, 255], 2);
        cv.line(dst, {x: cx, y: cy - 15}, {x: cx, y: cy + 15}, [255, 0, 255, 255], 2);
    }
    
    // ========== MARKER-LESS TRACKING (Template Matching) ==========
    setTemplate(rect, frame) {
        try {
            // Cleanup old template
            if (this.template) {
                try { this.template.delete(); } catch(e) {}
            }
            
            // Clamp coordinates
            const x = Math.max(0, Math.floor(rect.x));
            const y = Math.max(0, Math.floor(rect.y));
            const w = Math.min(Math.floor(rect.width), frame.cols - x);
            const h = Math.min(Math.floor(rect.height), frame.rows - y);
            
            if (w < 10 || h < 10) {
                console.warn('Template too small');
                return;
            }
            
            // Extract template region
            const templateRect = new cv.Rect(x, y, w, h);
            this.template = frame.roi(templateRect).clone();
            this.templateRect = { x, y, width: w, height: h };
            
            console.log(`Template set: ${w}x${h} at (${x}, ${y})`);
        } catch (e) {
            console.error('Error setting template:', e);
            this.template = null;
        }
    }
    
    trackMarkerless(src, dst) {
        if (!this.template || this.template.empty()) {
            return false;
        }
        
        try {
            // Convert both to grayscale
            const srcGray = new cv.Mat();
            const templateGray = new cv.Mat();
            
            cv.cvtColor(src, srcGray, cv.COLOR_RGBA2GRAY);
            cv.cvtColor(this.template, templateGray, cv.COLOR_RGBA2GRAY);
            
            // Check template fits in source
            if (templateGray.rows > srcGray.rows || templateGray.cols > srcGray.cols) {
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
            
            // Cleanup intermediate mats
            srcGray.delete();
            templateGray.delete();
            result.delete();
            
            // Threshold for valid detection
            if (confidence > 0.5) {
                const w = this.template.cols;
                const h = this.template.rows;
                
                // Draw detection
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
        const pt1 = {x: x, y: y};
        const pt2 = {x: x + w, y: y + h};
        cv.rectangle(dst, pt1, pt2, [0, 255, 0, 255], 3);
        
        // Cyan corners
        const corners = [
            {x: x, y: y},
            {x: x + w, y: y},
            {x: x + w, y: y + h},
            {x: x, y: y + h}
        ];
        corners.forEach(c => {
            cv.circle(dst, c, 6, [0, 255, 255, 255], -1);
        });
        
        // White center with green crosshair
        const cx = Math.round(x + w / 2);
        const cy = Math.round(y + h / 2);
        cv.circle(dst, {x: cx, y: cy}, 8, [255, 255, 255, 255], -1);
        cv.line(dst, {x: cx - 20, y: cy}, {x: cx + 20, y: cy}, [0, 255, 0, 255], 2);
        cv.line(dst, {x: cx, y: cy - 20}, {x: cx, y: cy + 20}, [0, 255, 0, 255], 2);
        
        // Confidence text
        const confText = `${Math.round(confidence * 100)}%`;
        cv.putText(dst, confText, {x: x, y: y - 10}, cv.FONT_HERSHEY_SIMPLEX, 0.7, [0, 255, 0, 255], 2);
    }
    
    // ========== SAM2 SEGMENTATION TRACKING ==========
    async loadSAM2Data(npzData) {
        try {
            this.sam2Data = npzData;
            this.sam2Masks = [];
            this.sam2Centroids = [];
            
            // Parse NPZ file
            const parser = new NPZParser();
            const npz = await parser.parseNPZ(npzData);
            
            console.log('NPZ parsed, keys:', Object.keys(npz));
            
            // Extract masks
            if (npz.masks) {
                const maskArray = npz.masks;
                console.log('Masks shape:', maskArray.shape);
                
                if (maskArray.shape.length === 3) {
                    // Multiple masks (N, H, W)
                    this.sam2Masks = parser.numpyToMat(maskArray);
                } else if (maskArray.shape.length === 2) {
                    // Single mask (H, W)
                    this.sam2Masks = [parser.numpyToMat(maskArray)];
                }
                
                // Compute centroids
                this.sam2Centroids = this.sam2Masks.map(mask => this.computeCentroid(mask)).filter(c => c);
                
                console.log(`Loaded ${this.sam2Masks.length} masks, ${this.sam2Centroids.length} centroids`);
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
            // Show placeholder if no masks loaded
            return this.trackSAM2Placeholder(src, dst);
        }
        
        let count = 0;
        
        try {
            for (let i = 0; i < this.sam2Masks.length; i++) {
                const mask = this.sam2Masks[i];
                const centroid = this.sam2Centroids[i];
                
                if (!mask || mask.empty()) continue;
                
                // Scale mask if needed
                let scaledMask = mask;
                if (mask.rows !== src.rows || mask.cols !== src.cols) {
                    scaledMask = new cv.Mat();
                    cv.resize(mask, scaledMask, new cv.Size(src.cols, src.rows));
                }
                
                // Get bounding rect
                const rect = cv.boundingRect(scaledMask);
                
                // Draw magenta bounding box
                const pt1 = {x: rect.x, y: rect.y};
                const pt2 = {x: rect.x + rect.width, y: rect.y + rect.height};
                cv.rectangle(dst, pt1, pt2, [255, 0, 255, 255], 3);
                
                // Draw centroid
                if (centroid) {
                    const cx = Math.round(centroid.x * (src.cols / mask.cols));
                    const cy = Math.round(centroid.y * (src.rows / mask.rows));
                    cv.circle(dst, {x: cx, y: cy}, 10, [255, 0, 255, 255], -1);
                    cv.line(dst, {x: cx - 20, y: cy}, {x: cx + 20, y: cy}, [255, 0, 255, 255], 2);
                    cv.line(dst, {x: cx, y: cy - 20}, {x: cx, y: cy + 20}, [255, 0, 255, 255], 2);
                }
                
                if (scaledMask !== mask) {
                    scaledMask.delete();
                }
                
                count++;
            }
        } catch (e) {
            console.error('SAM2 tracking error:', e);
        }
        
        return { found: count > 0, count };
    }
    
    trackSAM2Placeholder(src, dst) {
        // Show placeholder when no SAM2 data loaded
        const x = Math.floor(src.cols * 0.25);
        const y = Math.floor(src.rows * 0.25);
        const w = Math.floor(src.cols * 0.5);
        const h = Math.floor(src.rows * 0.5);
        
        // Dashed rectangle effect
        const pt1 = {x: x, y: y};
        const pt2 = {x: x + w, y: y + h};
        cv.rectangle(dst, pt1, pt2, [255, 0, 255, 255], 2);
        
        // Center text
        const cx = Math.round(x + w / 2);
        const cy = Math.round(y + h / 2);
        cv.circle(dst, {x: cx, y: cy}, 10, [255, 0, 255, 255], -1);
        cv.putText(dst, "Load NPZ", {x: cx - 50, y: cy + 5}, cv.FONT_HERSHEY_SIMPLEX, 0.6, [255, 0, 255, 255], 2);
        
        return { found: false, count: 0 };
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
