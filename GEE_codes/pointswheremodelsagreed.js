// ============================================================
// Model Agreement Analysis — Mangrove Classification
// Identifies certain (high-confidence) pixels across 5 base
// learners, masks water, and exports stratified sample points
// ============================================================

// --- AOI and base learner probability images ---
var aoi = ee.FeatureCollection("projects/ee-islamkm/assets/bfd_range_aoi");

var models = {
  knn:    ee.Image("projects/ee-islamkm/assets/baselearner_knn_mngrv").rename('knn'),
  logreg: ee.Image("projects/ee-islamkm/assets/baselearner_logreg_mngrv").rename('logreg'),
  rf:     ee.Image("projects/ee-islamkm/assets/baselearner_rf_mngrv").rename('rf'),
  svc:    ee.Image("projects/ee-islamkm/assets/baselearner_svc_mgrv").rename('svc'),
  xgb:    ee.Image("projects/ee-islamkm/assets/baselearner_xgb_mgrv").rename('xgb')
};

var modelList = [models.knn, models.logreg, models.rf, models.svc, models.xgb];
var multiBandImage = models.knn.addBands([models.logreg, models.rf, models.svc, models.xgb]);
var palette = ['#0000FF','#0055FF','#0099FF','#FFFF00','#FF9900','#FF5500','#FF0000'];

// --- Map setup ---
Map.setCenter(89.5458, 22.0767, 9);
Map.addLayer(ee.ImageCollection(modelList).reduce(ee.Reducer.stdDev()),
  {min: 0, max: 0.2, palette: palette}, 'Standard Deviation', false);
Map.addLayer(multiBandImage, {}, 'Multi-band Image', false);
print('Multi-band Image:', multiBandImage);

// ============================================================
// LOW AGREEMENT ANALYSIS
// Binarize predictions (threshold = 0.5), retain only pixels
// where models disagree, compute per-pixel agreement (%)
// ============================================================

var binImage = multiBandImage.gt(0.5);
var binMasked = binImage.updateMask(binImage.reduce(ee.Reducer.countDistinct()).gt(1));

var agreementPercent = binMasked.reduce(ee.Reducer.sum())
  .divide(5).multiply(100).rename('agreementPercent');
var lowAgreementMasked = agreementPercent.lt(60).selfMask();

Map.addLayer(agreementPercent,
  {min: 0, max: 100, palette: ['blue','yellow','red']}, 'Classification Agreement (%)', false);
Map.addLayer(lowAgreementMasked,
  {min: 0, max: 0.98, palette: palette}, 'Low Agreement Areas (<60%)', false);

// ============================================================
// PATCH FILTERING — Remove isolated pixels
// Connected components with 7x7 kernel; visualize patch sizes
// ============================================================

var labels = lowAgreementMasked.connectedComponents({
  connectedness: ee.Kernel.square(3), maxSize: 1000
}).select('labels');

var objectSize = labels.connectedPixelCount({maxSize: 100, eightConnected: true});
Map.addLayer(objectSize,
  {min: 1, max: 100, palette: ['blue','cyan','green','yellow','red']}, 'Object Size (pixels)', false);

// ============================================================
// HIGH-CONFIDENCE PIXEL CLASSIFICATION
// allHigh: all models >= 0.7 → mangrove (class 1)
// allLow:  all models <= 0.3 → non-mangrove (class 0)
// ============================================================

var bands = ['knn','logreg','rf','svc','xgb'];
var allHigh = multiBandImage.select(bands).gte(0.7).reduce(ee.Reducer.min());
var allLow  = multiBandImage.select(bands).lte(0.3).reduce(ee.Reducer.min());

// Class image: 1 = certain mangrove, 0 = certain non-mangrove
var classImage = allHigh
  .where(allLow.eq(1), 0)
  .updateMask(allHigh.or(allLow))
  .rename('high_vs_low')
  .clip(aoi);

Map.addLayer(classImage, {min: 0, max: 1, palette: ['red','green']},
  'All bands >=0.7 (green) / <=0.3 (red)');

// ============================================================
// WATER MASKING — Dynamic World V1 (2020-06 to 2021-07)
// Pixels with median water probability > 0.1 are excluded
// ============================================================

var waterProb = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterDate('2020-06-01', '2021-07-31')
  .median()
  .select('water')
  .clip(aoi);

Map.addLayer(waterProb.gt(0.4), {min: 0, max: 1}, 'Water > 0.4', false);

// Combine class image with water band, then mask out water pixels (water > 0.1)
var maskedCombined = classImage.addBands(waterProb)
  .updateMask(waterProb.lte(0.1));

print('maskedCombined:', maskedCombined);
Map.addLayer(maskedCombined, {bands: ['high_vs_low'], min: 0, max: 1}, 'Masked high_vs_low');

// ============================================================
// STRATIFIED RANDOM POINT SAMPLING
// 100 pts per class (mangrove / non-mangrove), water excluded
// ============================================================

var poolSize  = 50000;  // large pool ensures enough pts per class
var nPerClass = 100;
var scale     = 4.77;   // match your image resolution (meters)

// Sample the class band across a random point pool
var sampled = maskedCombined.select('high_vs_low').reduceRegions({
  collection: ee.FeatureCollection.randomPoints(maskedCombined.geometry(), poolSize, /*seed=*/2),
  reducer: ee.Reducer.first().setOutputs(['high_vs_low']),
  scale: scale
}).filter(ee.Filter.notNull(['high_vs_low']));

var pointsClass1 = sampled.filter(ee.Filter.eq('high_vs_low', 1)).limit(nPerClass);
var pointsClass0 = sampled.filter(ee.Filter.eq('high_vs_low', 0)).limit(nPerClass);

Map.addLayer(pointsClass1, {color: 'red'},  'Points — mangrove (class 1)');
Map.addLayer(pointsClass0, {color: 'blue'}, 'Points — non-mangrove (class 0)');

// ============================================================
// EXPORT — Certain mangrove sample points (class 1 only)
// ============================================================

Export.table.toAsset({
  collection: pointsClass1,
  description: 'export_certain_point_1val_frombaselearners',
  assetId: 'export_certain_point_1val_frombaselearners'
});
