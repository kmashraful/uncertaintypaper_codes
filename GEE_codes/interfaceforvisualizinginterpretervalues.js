// #############################
// Inter interpreter analysis split screen v2.2
/////////////////////////////////////////////////
// ----------------- USER ASSETS -----------------
var fc1_id = 'projects/ee-islamkm/assets/interpreter1_200pts';
var fc2_id = 'projects/ee-islamkm/assets/interpreter2_200pts';
var fc3_id = 'projects/ee-islamkm/assets/interpreter3_200pts';

// Load all three feature collections
var fc1 = ee.FeatureCollection(fc1_id);
var fc2 = ee.FeatureCollection(fc2_id);
var fc3 = ee.FeatureCollection(fc3_id);

// Load the multi-band image
var multiBandImage = ee.Image("projects/ee-islamkm/assets/BD_NICFI_RAWBands_indices_w2w");
var imageVisParam = {"opacity":0.97,"bands":["R","G","B"],"min":0.023633792252292295,"max":0.0852236811124915,"gamma":1.0090000000000001};
var nbrVisParam = {"opacity":0.97,"bands":["N","B","R"],"min":0.023633792252292295,"max":0.0852236811124915,"gamma":1.0090000000000001};

// ----------------------------------------
// Define a shared, fixed color palette map
// NEW COLOR SCHEME:
// 0 = blue, 0.3 = purple, 0.5 = red, 0.7 = light gray, 1.0 = green
// ----------------------------------------
var valPalette = {
  0.0: '#1f78b4',  // blue
  0.3: '#9467bd',  // purple
  0.5: '#e31a1c',  // red
  0.7: '#d9d9d9',  // light gray
  1.0: '#33a02c'   // green
};

// Store layer names for toggling
var layerRegistry = {
  leftMap: {},
  centerMap: {},
  rightMap: {}
};

// --------------------------------------
// Create three synchronized maps
// --------------------------------------
var leftMap = ui.Map();
var centerMap = ui.Map();
var rightMap = ui.Map();

// ----------------------------------------------------
// Function to add val_int layers to a specific map
// ----------------------------------------------------
function addValIntLayers(map, mapName, fc, labelPrefix) {
  var v0  = ee.Number(0.0);
  var v03 = ee.Number(0.3);
  var v05 = ee.Number(0.5);
  var v07 = ee.Number(0.7);
  var v1  = ee.Number(1.0);
  
  function onlyVal(v) {
    return fc.filter(ee.Filter.eq('val_int', v));
  }
  
  function styled(subFc, hex) {
    return subFc.style({
      color: hex,
      pointSize: 5,
      width: 2
    });
  }
  
  // Store layer information for later toggling
  var valIntValues = [
    {val: v0, valNum: 0.0, color: valPalette[0.0], label: '0'},
    {val: v03, valNum: 0.3, color: valPalette[0.3], label: '0.3'},
    {val: v05, valNum: 0.5, color: valPalette[0.5], label: '0.5'},
    {val: v07, valNum: 0.7, color: valPalette[0.7], label: '0.7'},
    {val: v1, valNum: 1.0, color: valPalette[1.0], label: '1'}
  ];
  
  valIntValues.forEach(function(item) {
    var layerName = labelPrefix + ' val_int = ' + item.label;
    var layer = styled(onlyVal(item.val), item.color);
    map.addLayer(layer, {}, layerName, true);
    
    // Store layer name and interpreter prefix
    if (!layerRegistry[mapName][item.valNum]) {
      layerRegistry[mapName][item.valNum] = [];
    }
    layerRegistry[mapName][item.valNum].push(layerName);
  });
}

// Add base layers
leftMap.addLayer(multiBandImage, imageVisParam, "RGB Composite", true);
centerMap.addLayer(multiBandImage, nbrVisParam, "NBR Composite", true);
rightMap.setOptions('SATELLITE');

// Add interpreter layers
addValIntLayers(leftMap, 'leftMap', fc1, 'Interpreter 1');
addValIntLayers(leftMap, 'leftMap', fc2, 'Interpreter 2');
addValIntLayers(leftMap, 'leftMap', fc3, 'Interpreter 3');

addValIntLayers(centerMap, 'centerMap', fc1, 'Interpreter 1');
addValIntLayers(centerMap, 'centerMap', fc2, 'Interpreter 2');
addValIntLayers(centerMap, 'centerMap', fc3, 'Interpreter 3');

addValIntLayers(rightMap, 'rightMap', fc1, 'Interpreter 1');
addValIntLayers(rightMap, 'rightMap', fc2, 'Interpreter 2');
addValIntLayers(rightMap, 'rightMap', fc3, 'Interpreter 3');

// Link all three maps together
var linker = ui.Map.Linker([leftMap, centerMap, rightMap]);

// Center all maps on all points
var all = fc1.merge(fc2).merge(fc3);
leftMap.centerObject(all, 11);

// --------------------------------------
// Function to toggle layers by val_int
// --------------------------------------
function toggleLayersByValInt(valInt, isVisible) {
  var maps = [
    {map: leftMap, name: 'leftMap'},
    {map: centerMap, name: 'centerMap'},
    {map: rightMap, name: 'rightMap'}
  ];
  
  maps.forEach(function(mapObj) {
    var layerNames = layerRegistry[mapObj.name][valInt];
    if (layerNames) {
      layerNames.forEach(function(layerName) {
        var layers = mapObj.map.layers();
        for (var i = 0; i < layers.length(); i++) {
          var layer = layers.get(i);
          if (layer.getName() === layerName) {
            layer.setShown(isVisible);
          }
        }
      });
    }
  });
}

// --------------------------------------
// Create control panel with checkboxes
// --------------------------------------
var controlPanel = ui.Panel({
  style: {
    position: 'top-left',
    padding: '8px',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    width: '250px'
  }
});

var controlTitle = ui.Label('Filter Points by val_int', {
  fontWeight: 'bold',
  fontSize: '14px',
  margin: '0 0 8px 0'
});
controlPanel.add(controlTitle);

var instructionLabel = ui.Label('Toggle visibility for all interpreters across all panels:', {
  fontSize: '11px',
  margin: '0 0 8px 0',
  whiteSpace: 'pre-wrap'
});
controlPanel.add(instructionLabel);

// Create checkboxes for each val_int value with NEW COLORS
var valIntOptions = [
  {value: 0.0, label: 'val_int = 0 (Blue)', color: valPalette[0.0]},
  {value: 0.3, label: 'val_int = 0.3 (Purple)', color: valPalette[0.3]},
  {value: 0.5, label: 'val_int = 0.5 (Red)', color: valPalette[0.5]},
  {value: 0.7, label: 'val_int = 0.7 (Light Gray)', color: valPalette[0.7]},
  {value: 1.0, label: 'val_int = 1 (Green)', color: valPalette[1.0]}
];

valIntOptions.forEach(function(option) {
  var checkboxPanel = ui.Panel({
    layout: ui.Panel.Layout.flow('horizontal'),
    style: {margin: '2px 0'}
  });
  
  var colorBox = ui.Label('●', {
    color: option.color,
    fontSize: '16px',
    margin: '0 4px 0 0'
  });
  
  var checkbox = ui.Checkbox({
    label: option.label,
    value: true,
    style: {stretch: 'horizontal'}
  });
  
  checkbox.onChange(function(checked) {
    toggleLayersByValInt(option.value, checked);
  });
  
  checkboxPanel.add(colorBox);
  checkboxPanel.add(checkbox);
  controlPanel.add(checkboxPanel);
});

// Add separator
controlPanel.add(ui.Label('______________________________', {margin: '8px 0', fontSize: '10px'}));

// Quick filter buttons
var quickFilterTitle = ui.Label('Quick Filters:', {
  fontWeight: 'bold',
  fontSize: '12px',
  margin: '4px 0'
});
controlPanel.add(quickFilterTitle);

var showAllButton = ui.Button({
  label: 'Show All',
  style: {stretch: 'horizontal'}
});
showAllButton.onClick(function() {
  valIntOptions.forEach(function(option) {
    toggleLayersByValInt(option.value, true);
  });
  // Update checkbox states
  var widgets = controlPanel.widgets();
  for (var i = 0; i < widgets.length(); i++) {
    var widget = widgets.get(i);
    if (widget.widgets) {
      var subWidgets = widget.widgets();
      for (var j = 0; j < subWidgets.length(); j++) {
        var subWidget = subWidgets.get(j);
        if (subWidget.getLabel && subWidget.setValue) {
          subWidget.setValue(true, false);
        }
      }
    }
  }
});

var hideAllButton = ui.Button({
  label: 'Hide All',
  style: {stretch: 'horizontal'}
});
hideAllButton.onClick(function() {
  valIntOptions.forEach(function(option) {
    toggleLayersByValInt(option.value, false);
  });
  // Update checkbox states
  var widgets = controlPanel.widgets();
  for (var i = 0; i < widgets.length(); i++) {
    var widget = widgets.get(i);
    if (widget.widgets) {
      var subWidgets = widget.widgets();
      for (var j = 0; j < subWidgets.length(); j++) {
        var subWidget = subWidgets.get(j);
        if (subWidget.getLabel && subWidget.setValue) {
          subWidget.setValue(false, false);
        }
      }
    }
  }
});

var highValuesButton = ui.Button({
  label: 'Show Only High (0.7 & 1.0)',
  style: {stretch: 'horizontal'}
});
highValuesButton.onClick(function() {
  toggleLayersByValInt(0.0, false);
  toggleLayersByValInt(0.3, false);
  toggleLayersByValInt(0.5, false);
  toggleLayersByValInt(0.7, true);
  toggleLayersByValInt(1.0, true);
});

controlPanel.add(showAllButton);
controlPanel.add(hideAllButton);
controlPanel.add(highValuesButton);

// --------------------------------------
// Create titles for each panel
// --------------------------------------
var leftTitle = ui.Label('RGB Composite', {
  fontWeight: 'bold',
  fontSize: '16px',
  padding: '8px',
  textAlign: 'center',
  stretch: 'horizontal',
  backgroundColor: 'rgba(255, 255, 255, 0.8)'
});

var centerTitle = ui.Label('NBR Composite', {
  fontWeight: 'bold',
  fontSize: '16px',
  padding: '8px',
  textAlign: 'center',
  stretch: 'horizontal',
  backgroundColor: 'rgba(255, 255, 255, 0.8)'
});

var rightTitle = ui.Label('Satellite Basemap', {
  fontWeight: 'bold',
  fontSize: '16px',
  padding: '8px',
  textAlign: 'center',
  stretch: 'horizontal',
  backgroundColor: 'rgba(255, 255, 255, 0.8)'
});

// --------------------------------------
// Create the layout with three maps side by side
// --------------------------------------
var leftPanel = ui.Panel({
  widgets: [leftTitle, leftMap],
  style: {
    width: '33.33%',
    height: '100%'
  }
});

var centerPanel = ui.Panel({
  widgets: [centerTitle, centerMap],
  style: {
    width: '33.33%',
    height: '100%'
  }
});

var rightPanel = ui.Panel({
  widgets: [rightTitle, rightMap],
  style: {
    width: '33.33%',
    height: '100%'
  }
});

// Create horizontal panel containing all three
var mainPanel = ui.Panel({
  widgets: [leftPanel, centerPanel, rightPanel],
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {stretch: 'both'}
});

// Reset the root and add the triple split panel
ui.root.clear();
ui.root.add(mainPanel);

// Add control panel to the left map
leftMap.add(controlPanel);

// --------------------------------------
// Optional: print counts for sanity check
// --------------------------------------
print('FC1 total points:', fc1.size());
print('FC2 total points:', fc2.size());
print('FC3 total points:', fc3.size());

print('FC1 breakdown by val_int:',
  ee.Dictionary({
    '0'  : fc1.filter(ee.Filter.eq('val_int', 0.0)).size(),
    '0.3': fc1.filter(ee.Filter.eq('val_int', 0.3)).size(),
    '0.5': fc1.filter(ee.Filter.eq('val_int', 0.5)).size(),
    '0.7': fc1.filter(ee.Filter.eq('val_int', 0.7)).size(),
    '1'  : fc1.filter(ee.Filter.eq('val_int', 1.0)).size()
  })
);

print('FC2 breakdown by val_int:',
  ee.Dictionary({
    '0'  : fc2.filter(ee.Filter.eq('val_int', 0.0)).size(),
    '0.3': fc2.filter(ee.Filter.eq('val_int', 0.3)).size(),
    '0.5': fc2.filter(ee.Filter.eq('val_int', 0.5)).size(),
    '0.7': fc2.filter(ee.Filter.eq('val_int', 0.7)).size(),
    '1'  : fc2.filter(ee.Filter.eq('val_int', 1.0)).size()
  })
);

print('FC3 breakdown by val_int:',
  ee.Dictionary({
    '0'  : fc3.filter(ee.Filter.eq('val_int', 0.0)).size(),
    '0.3': fc3.filter(ee.Filter.eq('val_int', 0.3)).size(),
    '0.5': fc3.filter(ee.Filter.eq('val_int', 0.5)).size(),
    '0.7': fc3.filter(ee.Filter.eq('val_int', 0.7)).size(),
    '1'  : fc3.filter(ee.Filter.eq('val_int', 1.0)).size()
  })
);
