// ============================================================
// Model Disagreement Analysis — Mangrove Classification
// Compares 5 base learners and identifies low-agreement zones
// ============================================================

// --- Load base learner probability images ---
var models = {
  knn:    ee.Image("projects/ee-islamkm/assets/baselearner_knn_mngrv").rename('knn'),
  logreg: ee.Image("projects/ee-islamkm/assets/baselearner_logreg_mngrv").rename('logreg'),
  rf:     ee.Image("projects/ee-islamkm/assets/baselearner_rf_mngrv").rename('rf'),
  svc:    ee.Image("projects/ee-islamkm/assets/baselearner_svc_mgrv").rename('svc'),
  xgb:    ee.Image("projects/ee-islamkm/assets/baselearner_xgb_mgrv").rename('xgb')
};

var palette = ['#0000FF','#0055FF','#0099FF','#FFFF00','#FF9900','#FF5500','#FF0000'];
var modelList = [models.knn, models.logreg, models.rf, models.svc, models.xgb];
var modelCollection = ee.ImageCollection(modelList);
var multiBandImage = models.knn.addBands([models.logreg, models.rf, models.svc, models.xgb]);

// --- Map setup ---
Map.setCenter(89.5458, 22.0767, 9);
Map.addLayer(modelCollection.reduce(ee.Reducer.stdDev()),
  {min: 0, max: 0.2, palette: palette}, 'Standard Deviation', false);
Map.addLayer(multiBandImage, {}, 'Multi-band Image', false);
print('Multi-band Image:', multiBandImage);

// ============================================================
// LOW AGREEMENT ANALYSIS
// Binarize predictions (threshold = 0.5), then find pixels
// where models disagree on the mangrove/non-mangrove label
// ============================================================

// Binarize all models and stack into one multi-band image
var binImage = multiBandImage.gt(0.5);

// Keep only pixels with disagreement (at least two distinct values across models)
var disagreementMask = binImage.reduce(ee.Reducer.countDistinct()).gt(1);
var binMasked = binImage.updateMask(disagreementMask);

// Compute agreement percentage among disagreeing pixels
var agreementPercent = binMasked.reduce(ee.Reducer.sum())
  .divide(5).multiply(100).rename('agreementPercent');

// Mask to low-agreement pixels only (<60% agreement)
var lowAgreement = agreementPercent.lt(60);
var lowAgreementMasked = lowAgreement.updateMask(lowAgreement);

Map.addLayer(agreementPercent,
  {min: 0, max: 100, palette: ['blue','yellow','red']}, 'Classification Agreement (%)', false);
Map.addLayer(lowAgreementMasked,
  {min: 0, max: 0.98, palette: palette}, 'Low Agreement Areas (<60%)', false);

// ============================================================
// PATCH FILTERING — Remove isolated pixels
// Uses connected component labeling to retain only contiguous
// patches >= 50 pixels (7x7 kernel / 8-connectivity)
// ============================================================

// Label connected components using a 7x7 (radius=3) kernel
var labels = lowAgreementMasked.connectedComponents({
  connectedness: ee.Kernel.square(3),
  maxSize: 1000
}).select('labels');

// Filter to patches >= 50 pixels
var largePatchesMask = labels.connectedPixelCount({maxSize: 100, eightConnected: true}).gte(50);
var largePatches = labels.updateMask(largePatchesMask);

Map.addLayer(largePatches.connectedPixelCount({maxSize: 100, eightConnected: true}),
  {min: 1, max: 100, palette: ['blue','cyan','green','yellow','red']}, 'Object Size (pixels)', false);

// Vectorize large patches
var largePatchVectors = largePatches.reduceToVectors({
  geometryType: 'polygon',
  labelProperty: 'patch_id',
  reducer: ee.Reducer.countEvery(),
  geometry: lowAgreementMasked.geometry(),
  scale: 4, maxPixels: 1e10, bestEffort: true, tileScale: 5
});

// ============================================================
// EXPORT — 200 random sample points within uncertain patches
// ============================================================

Export.table.toAsset({
  collection: ee.FeatureCollection.randomPoints({
    region: largePatchVectors.geometry(),
    points: 200,
    seed: 2
  }),
  description: 'export_randomPoints_200pts',
  assetId: 'confued_sampledPoints200more_5models_3'
});
