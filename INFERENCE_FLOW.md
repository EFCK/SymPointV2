# İnferans Modülü - Input Akış Analizi

Bu dokümantasyon, inference modülünün çalıştığında input'un girişten çıkışa kadar olan yolunu, girdiği fonksiyonları, input shape'lerini ve ne olduğunu detaylı olarak açıklar.

## 1. ENTRY POINT: `tools/inference.py` → `main()`

**Fonksiyon**: `main()` (satır 55)
**Dosya**: `tools/inference.py`

### 1.1. Veri Dosyalarının Bulunması
```84:84:tools/inference.py
coords, feats, labels,lengths,layerIds = SVGDataset.load(svg_file,idx=1)
```
- `data_list`: JSON dosyalarının listesi (`*_s2.json` pattern'i ile)
- Her bir SVG dosyası için işlem başlar

---

## 2. VERİ YÜKLEME: `SVGDataset.load()`

**Fonksiyon**: `SVGDataset.load()` (statik method)
**Dosya**: `svgnet/data/svg3.py`
**Satır**: 88-146

### 2.1. Input
- **`json_file`**: JSON dosya yolu (örn: `dataset/json/floorplan_demo_wo_layer/0003-0010_s2.json`)
- **`idx`**: Instance ID offset için kullanılan indeks (inference'da 1)

### 2.2. JSON'dan Veri Okuma (satır 90-92)
```python
data = json.load(open(json_file))
width, height = data["width"], data["height"]  # SVG genişlik/yükseklik
coords = data["args"]  # Koordinat array'i
```

### 2.3. Koordinat İşleme (satır 93-96)
```93:96:svgnet/data/svg3.py
arcs = angles_with_horizontal(coords)
coords = np.array(coords).reshape(-1,8)
coords[:, 0::2] = coords[:, 0::2] / width   # x koordinatlarını normalize et
coords[:, 1::2] = coords[:, 1::2] / height  # y koordinatlarını normalize et
```

**Shape Dönüşümü**:
- `coords`: `(N, 8)` → Her satır bir SVG segmentini temsil eder (4 nokta × 2 koordinat)
- Normalize edilmiş koordinatlar [0,1] aralığında

### 2.4. Point Cloud Oluşturma (satır 98-108)
```101:108:svgnet/data/svg3.py
coord = np.zeros((max_num,3))
coord_x = np.mean(coords[:,0::2],axis=1) 
coord_y = np.mean(coords[:,1::2],axis=1) 
coord_z = np.zeros((num,))

coord[:num,0] = coord_x
coord[:num,1] = coord_y
coord[:num,2] = coord_z
```

**Shape**:
- `coord`: `(max_num, 3)` → 3D point cloud (x, y, z=0)
- Her segment'in orta noktası alınır
- `max_num = max(num, min_points)` → minimum 2048 nokta

### 2.5. Feature Vektörü Oluşturma (satır 113-124)
```113:124:svgnet/data/svg3.py
feat = np.zeros((max_num,7))
max_lengths = max(width, height)
lens = np.array(data["lengths"]).clip(0,max_lengths) / max_lengths
ctype = np.eye(4)[data["commands"]]
widths = np.array(data["widths"])
feat[:num,0] = arcs
feat[:num,1] = lens
feat[:num,2:6] = ctype
feat[:num,6] = widths/np.max(widths)
```

**Feature Kanalları** (7 boyut):
- `feat[:, 0]`: `arcs` - Segment'in yatayla açısı (radyan)
- `feat[:, 1]`: `lens` - Normalize edilmiş segment uzunluğu
- `feat[:, 2:6]`: `ctype` - SVG komut tipi (one-hot encoding, 4 tip: M, L, C, Z)
- `feat[:, 6]`: `widths` - Normalize edilmiş çizgi kalınlığı

**Shape**:
- `feat`: `(max_num, 7)`

### 2.6. Label Oluşturma (satır 127-139)
```127:139:svgnet/data/svg3.py
semanticIds = np.full_like(coord[:,0],35) # bg sem id = 35
seg = np.array(data["semanticIds"])
semanticIds[:num] = seg
semanticIds = semanticIds.astype(np.int64)

instanceIds = np.full_like(coord[:,0],-1) # stuff id = -1
ins = np.array(data["instanceIds"])
valid_pos = ins != -1
ins[valid_pos] += idx*min_points

instanceIds[:num] = ins
instanceIds = instanceIds.astype(np.int64)
label = np.concatenate([semanticIds[:,None],instanceIds[:,None]],axis=1)
```

**Shape**:
- `label`: `(max_num, 2)`
  - `label[:, 0]`: Semantic ID (1-36 arası, 35=background)
  - `label[:, 1]`: Instance ID (-1 = stuff/background, >=0 = thing instances)

### 2.7. Layer ID (satır 141-145)
```141:145:svgnet/data/svg3.py
layerIds = np.array(data["layerIds"])
bg_layerId = np.max(layerIds)+1
pad_layerIds = np.full_like(coord[:,0],bg_layerId)
pad_layerIds[:num] = layerIds
pad_layerIds += idx*min_points
```

**Shape**:
- `layerIds`: `(max_num,)` → Her noktanın hangi layer'da olduğu

### 2.8. Return Değerleri
```146:146:svgnet/data/svg3.py
return coord, feat, label,lengths,pad_layerIds
```

**Return Types**:
- `coord`: `(max_num, 3)` - numpy float array
- `feat`: `(max_num, 7)` - numpy float array
- `label`: `(max_num, 2)` - numpy int64 array
- `lengths`: `(max_num,)` - numpy float array (segment uzunlukları)
- `pad_layerIds`: `(max_num,)` - numpy array

---

## 3. BATCH HAZIRLAMA: `tools/inference.py` → `main()` (satır 84-90)

### 3.1. Koordinat Normalizasyonu (satır 85)
```85:85:tools/inference.py
coords -= np.mean(coords, 0)
```
- Koordinatlar merkeze hizalanır (mean subtraction)

### 3.2. Offset Hesaplama (satır 86)
```86:86:tools/inference.py
offset = [coords.shape[0]]
```
- Batch'teki nokta sayısını içeren liste
- **Shape**: `[N]` (N = nokta sayısı)

### 3.3. Torch Tensor'a Dönüştürme (satır 87-89)
```87:89:tools/inference.py
layerIds = torch.LongTensor(layerIds)
offset = torch.IntTensor(offset)
coords,feats,labels = torch.FloatTensor(coords), torch.FloatTensor(feats), torch.LongTensor(labels)
```

### 3.4. Batch Tuple Oluşturma (satır 90)
```90:90:tools/inference.py
batch = (coords,feats,labels,offset, torch.FloatTensor(lengths),layerIds)
```

**Final Batch Shape'leri**:
- `coords`: `torch.FloatTensor` → `(N, 3)` - 3D koordinatlar
- `feats`: `torch.FloatTensor` → `(N, 7)` - Feature vektörleri
- `labels`: `torch.LongTensor` → `(N, 2)` - [semantic_id, instance_id]
- `offset`: `torch.IntTensor` → `(1,)` - Batch offset
- `lengths`: `torch.FloatTensor` → `(N,)` - Segment uzunlukları
- `layerIds`: `torch.LongTensor` → `(N,)` - Layer ID'leri

**Örnek**: Eğer 5000 nokta varsa:
- `coords`: `(5000, 3)`
- `feats`: `(5000, 7)`
- `labels`: `(5000, 2)`
- `offset`: `[5000]`
- `lengths`: `(5000,)`
- `layerIds`: `(5000,)`

---

## 4. MODEL FORWARD: `svgnet/model/svgnet.py` → `forward()`

**Fonksiyon**: `SVGNet.forward()` (satır 29-31)
**Dosya**: `svgnet/model/svgnet.py`

### 4.1. Batch Unpacking (satır 30)
```30:30:svgnet/model/svgnet.py
coords,feats,semantic_labels,offsets,lengths,layerIds = batch
```

### 4.2. Internal Forward Call (satır 31)
```31:31:svgnet/model/svgnet.py
return self._forward(coords,feats,offsets,semantic_labels,lengths,layerIds,return_loss=return_loss)
```

---

## 5. INTERNAL FORWARD: `svgnet/model/svgnet.py` → `_forward()`

**Fonksiyon**: `SVGNet._forward()` (satır 69-128)
**Dosya**: `svgnet/model/svgnet.py`

### 5.1. Stage List Oluşturma (satır 80)
```80:80:svgnet/model/svgnet.py
stage_list={'inputs': {'p_out':coords,"f_out":feats,"offset":offsets},"semantic_labels":semantic_labels[:,0]}
```

**Stage List İçeriği**:
- `stage_list['inputs']['p_out']`: `(N, 3)` - Koordinatlar
- `stage_list['inputs']['f_out']`: `(N, 7)` - Feature'lar
- `stage_list['inputs']['offset']`: `(1,)` - Batch offset
- `stage_list['semantic_labels']`: `(N,)` - Sadece semantic ID'ler (label'ın ilk sütunu)

### 5.2. Target Hazırlama (satır 81)
```81:81:svgnet/model/svgnet.py
targets = self.prepare_targets(semantic_labels)
```

**Fonksiyon**: `prepare_targets()` (satır 33-66)

Bu fonksiyon:
- Her unique `(semantic_id, instance_id)` çifti için bir target mask oluşturur
- Background (sem_id=35, ins_id=-1) skip edilir

**Return**: 
```python
[{
    "labels": torch.Tensor (K,) - K adet semantic ID
    "masks": torch.Tensor (N, K) - Her instance için binary mask
}]
```

**Shape**:
- `targets[0]["labels"]`: `(K,)` - K = unique instance sayısı ( kapı1, kapı2)
- `targets[0]["masks"]`: `(N, K)` - Binary mask matrix

### 5.3. Backbone İşleme (satır 84)
```84:84:svgnet/model/svgnet.py
stage_list = self.backbone(stage_list)
```

**Backbone**: `PointTransformer` (`svgnet/model/pointtransformer.py`)

---

## 6. BACKBONE İŞLEME: `svgnet/model/pointtransformer.py` → `forward()`

**Fonksiyon**: `PointTransformer.forward()` (satır 45-89)
**Dosya**: `svgnet/model/pointtransformer.py`

### 6.1. Input Extraction (satır 47-49)
```47:49:svgnet/model/pointtransformer.py
p0 = stage_list["inputs"]["p_out"] # (n, 3)
x0 = stage_list["inputs"]["f_out"] # (n, c), 
o0 = stage_list["inputs"]["offset"]#  (b)
```

**Shape'ler**:
- `p0`: `(N, 3)` - Koordinatlar
- `x0`: `(N, 7)` - Feature'lar (7 kanal)
- `o0`: `(1,)` - Batch offset

### 6.2. Feature Concatenation (satır 51)
```51:51:svgnet/model/pointtransformer.py
x0 = p0 if self.in_planes == 3 else torch.cat((p0, x0), 1)
```

**Sonuç**: 
- `x0`: `(N, 10)` - `p0` (3) + `x0` (7) = 10 kanal

### 6.3. Encoder Layers (satır 52-64)

#### Encoder 1 (satır 52-53)
```52:53:svgnet/model/pointtransformer.py
[p1, x1, o1] = self.enc1[0]([p0, x0, o0])
p1, x1, o1, knn_idx1 = self.enc1[1:]([p1, x1, o1,None])
```
- **Output**: 
  - `p1`: `(N, 3)` - Aynı nokta sayısı
  - `x1`: `(N, 64)` - Feature dimension 64'e çıkarıldı

#### Encoder 2 (satır 54-55)
```54:55:svgnet/model/pointtransformer.py
[p2, x2, o2], idx2 = self.enc2[0]([p1, x1, o1])
p2, x2, o2,knn_idx2  = self.enc2[1:]([p2, x2, o2, None])
```
- **Output**: 
  - `p2`: `(N/4, 3)` - Downsampled (1/4)
  - `x2`: `(N/4, 128)` - Feature dimension 128

#### Encoder 3 (satır 57-58)
```57:58:svgnet/model/pointtransformer.py
[p3, x3, o3], idx3 = self.enc3[0]([p2, x2, o2])
p3, x3, o3,knn_idx3 = self.enc3[1:]([p3, x3, o3, None])
```
- **Output**: 
  - `p3`: `(N/16, 3)` - Downsampled (1/16)
  - `x3`: `(N/16, 256)` - Feature dimension 256

#### Encoder 4 (satır 60-61)
```60:61:svgnet/model/pointtransformer.py
[p4, x4, o4], idx4 = self.enc4[0]([p3, x3, o3])
p4, x4, o4,knn_idx4 = self.enc4[1:]([p4, x4, o4, None])
```
- **Output**: 
  - `p4`: `(N/64, 3)` - Downsampled (1/64)
  - `x4`: `(N/64, 512)` - Feature dimension 512

#### Encoder 5 (satır 63-64)
```63:64:svgnet/model/pointtransformer.py
[p5, x5, o5], idx5 = self.enc5[0]([p4, x4, o4])
p5, x5, o5,knn_idx5 = self.enc5[1:]([p5, x5, o5,None])
```
- **Output**: 
  - `p5`: `(N/256, 3)` - Downsampled (1/256)
  - `x5`: `(N/256, 1024)` - Feature dimension 1024

### 6.4. Decoder Layers (satır 75-79)

#### Decoder 5 → 4 (satır 75-76)
```75:76:svgnet/model/pointtransformer.py
x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5,knn_idx5])[1]
x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4,knn_idx4])[1]
```
- **Output**: 
  - `x4`: `(N/64, 512)` - Upsampled features

#### Decoder 4 → 3 (satır 77)
```77:77:svgnet/model/pointtransformer.py
x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3,knn_idx3])[1]
```
- **Output**: 
  - `x3`: `(N/16, 256)` - Upsampled features

#### Decoder 3 → 2 (satır 78)
```78:78:svgnet/model/pointtransformer.py
x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2,knn_idx2])[1]
```
- **Output**: 
  - `x2`: `(N/4, 128)` - Upsampled features

#### Decoder 2 → 1 (satır 79)
```79:79:svgnet/model/pointtransformer.py
x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1,knn_idx1])[1]
```
- **Output**: 
  - `x1`: `(N, 64)` - Full resolution features

### 6.5. Stage List Update (satır 81-88)
```81:88:svgnet/model/pointtransformer.py
up_list = [
    {'p_out': p1, 'f_out': x1, 'offset': o1},  # n_0 = n, fdims = 32
    {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
    {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
    {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3
    {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4 - fdims = 512 (extracted through dec5 = mlps)
]
stage_list['up'] = up_list
```

**Up List Shape'leri** (N=5000 örneği için):
- `up_list[0]`: `p1`: `(5000, 3)`, `x1`: `(5000, 64)`
- `up_list[1]`: `p2`: `(1250, 3)`, `x2`: `(1250, 128)`
- `up_list[2]`: `p3`: `(312, 3)`, `x3`: `(312, 256)`
- `up_list[3]`: `p4`: `(78, 3)`, `x4`: `(78, 512)`
- `up_list[4]`: `p5`: `(19, 3)`, `x5`: `(19, 1024)`

---

## 7. DECODER İŞLEME: `svgnet/model/decoder.py` → `forward()`

**Fonksiyon**: `Decoder.forward()` (satır 272-365)
**Dosya**: `svgnet/model/decoder.py`

### 7.1. Query İnit (satır 274-276)
```274:276:svgnet/model/decoder.py
queries = self.query_feat.weight[:,None,:]
query_pos = self.query_pos.weight[:,None,:]
```

**Shape'ler**:
- `queries`: `(num_queries, 1, hidden_dim)` - Öğrenilmiş query embeddings
- `query_pos`: `(num_queries, 1, hidden_dim)` - Query position embeddings

### 7.2. Layer Feature Fusion (satır 294)
```294:294:svgnet/model/decoder.py
fusion_features = self.fusionLayerFeats(srcs[-1], layerIds)
```

**Fonksiyon**: `fusionLayerFeats()` (satır 110-137)
- Layer ID'lere göre feature'ları birleştirir
- Her layer için: average pooling + max pooling + attention pooling

**Input**:
- `srcs[-1]`: `(N, 64)` - Encoder'dan gelen feature'lar
- `layerIds`: `(N,)` - Layer ID'leri

**Output**:
- `fusion_features`: `(N, 64)` - Layer-aware feature'lar

### 7.3. Mask Features Head (satır 295)
```295:295:svgnet/model/decoder.py
mask_features = self.mask_features_head(fusion_features)
```

**Output**:
- `mask_features`: `(N, hidden_dim)` - Mask prediction için feature'lar

### 7.4. Decoder Layers (satır 301-336)

Her decoder layer'da:
1. **Mask Module** (satır 308-312): Query'lerden class ve mask prediction
2. **Cross Attention** (satır 315-322): Query'ler ile point cloud feature'ları arasında attention
3. **Self Attention** (satır 324-328): Query'ler arasında self-attention
4. **FFN** (satır 330-332): Feed-forward network

**Output Shape'leri**:
- `predictions_class`: List of `(1, num_queries, num_classes+1)`
- `predictions_mask`: List of `(1, num_queries, N)`

### 7.5. Final Mask Module (satır 337)
```337:337:svgnet/model/decoder.py
output_class, outputs_mask = self.mask_module(queries,mask_features)
```

**Final Output**:
- `output_class`: `(num_queries, num_classes+1)` - Class predictions
- `outputs_mask`: `(num_queries, N)` - Mask predictions (her query için tüm noktalara ait mask)

### 7.6. Return (satır 355-365)
```355:365:svgnet/model/decoder.py
out = {
    'pred_logits': predictions_class[-1],  # (1, num_queries, num_classes+1)
    'pred_masks': predictions_mask[-1],    # (1, num_queries, N)
    'aux_outputs': self._set_aux_loss(
        predictions_class , predictions_mask
    ),
    "stage_list": stage_list,
    'dn_out': dn_out if self.training else None,
}
```

---

## 8. INFERENCE FONKSİYONLARI: `svgnet/model/svgnet.py` → `_forward()`

### 8.1. Semantic Inference (satır 89)
```89:89:svgnet/model/svgnet.py
semantic_scores=self.semantic_inference(outputs["pred_logits"],outputs["pred_masks"])
```

**Fonksiyon**: `semantic_inference()` (satır 131-136)
```131:136:svgnet/model/svgnet.py
def semantic_inference(self, mask_cls, mask_pred):
    
    mask_cls = F.softmax(mask_cls, dim=-1)[...,:-1] # Q,C
    mask_pred = mask_pred.sigmoid() # Q,G
    semseg = torch.einsum("bqc,bqg->bgc", mask_cls, mask_pred)
    return semseg[0]
```

**İşlem**:
1. `mask_cls`: `(1, num_queries, num_classes+1)` → Softmax → Background hariç → `(1, num_queries, num_classes)`
2. `mask_pred`: `(1, num_queries, N)` → Sigmoid → `(1, num_queries, N)`
3. Einstein summation: `bqc,bqg->bgc` → Her nokta için class probability'leri

**Output**:
- `semantic_scores`: `(N, num_classes)` - Her nokta için semantic class probability'leri

### 8.2. Instance Inference (satır 90)
```90:90:svgnet/model/svgnet.py
instances = self.instance_inference(outputs["pred_logits"],outputs["pred_masks"])
```

**Fonksiyon**: `instance_inference()` (satır 138-179)

**İşlem**:
1. Score ve label hesaplama (softmax + argmax)
2. Background ve düşük score'lu query'leri filtrele
3. Her query için mask oluşturma
4. Overlap threshold kontrolü

**Output**:
- `instances`: List of dictionaries
  ```python
  [
      {
          "masks": np.array (N,) - Binary mask
          "labels": int - Predicted class ID
          "scores": float - Confidence score
      },
      ...
  ]
  ```

---

## 9. OUTPUT COLLECTION: `tools/inference.py` → `main()`

### 9.1. Result Dictionary (satır 112-118)
```112:118:tools/inference.py
save_dicts.append({
    "filepath": svg_file.replace("dataset/json/", "dataset/svg/").replace("_s2.json",".svg"),
    "sem": res["semantic_scores"].cpu().numpy(),
    "ins": res["instances"],
    "targets":res["targets"],
    "lengths":res["lengths"],
})
```

**Output Dictionary İçeriği**:
- `filepath`: SVG dosya yolu
- `sem`: `(N, num_classes)` numpy array - Semantic scores
- `ins`: List of instance dictionaries
- `targets`: Ground truth targets (eğer varsa)
- `lengths`: `(N,)` numpy array - Segment uzunlukları

### 9.2. Saving (satır 122)
```122:122:tools/inference.py
np.save(osp.join(args.out, 'model_output.npy'), save_dicts)
```

**Final Output**: `model_output.npy` - Tüm sonuçlar bu dosyaya kaydedilir

---

## ÖZET: SHAPE DEĞİŞİM ŞEMASI

```
JSON Dosyası
    ↓
SVGDataset.load()
    ↓
coord: (N, 3)       feat: (N, 7)       label: (N, 2)      lengths: (N,)    layerIds: (N,)
    ↓
Batch Preparation (inference.py)
    ↓
batch = (coords, feats, labels, offset, lengths, layerIds)
    ↓
SVGNet.forward()
    ↓
SVGNet._forward()
    ↓
Backbone (PointTransformer)
    ↓
Encoder: (N, 10) → (N, 64) → (N/4, 128) → (N/16, 256) → (N/64, 512) → (N/256, 1024)
    ↓
Decoder: (N/256, 1024) → (N/64, 512) → (N/16, 256) → (N/4, 128) → (N, 64)
    ↓
Decoder (Transformer)
    ↓
pred_logits: (1, num_queries, num_classes+1)
pred_masks: (1, num_queries, N)
    ↓
semantic_inference()
    ↓
semantic_scores: (N, num_classes)
    ↓
instance_inference()
    ↓
instances: List[{masks: (N,), labels: int, scores: float}]
    ↓
Save to model_output.npy
```

---

## MASKELER NE İŞE YARIYOR?

Maskeler bu modelde **4 farklı amaçla** kullanılır:

### 1. **Target Masks (Ground Truth Masks)** 
**Yer**: `prepare_targets()` fonksiyonu
**Kullanım**: Training sırasında model öğrenirken ground truth instance'ları temsil eder

```33:66:svgnet/model/svgnet.py
def prepare_targets(self,semantic_labels,bg_ind=-1,bg_sem=35):
    # Her unique (semantic_id, instance_id) çifti için bir binary mask oluşturur
    # Örnek: "kapı1" instance'ı için → mask[:, k] = [0,0,1,1,1,0,0,...]
    # Mask'in 1 olduğu noktalar, o instance'a ait noktalardır
```

**Şekil**: `(N, K)` 
- N = toplam nokta sayısı
- K = unique instance sayısı
- Her sütun bir instance'ı temsil eder, o instance'a ait noktalar 1, diğerleri 0

**Örnek**:
```
Nokta 0: [0, 0, 1] → Nokta 0, instance 2'ye ait
Nokta 1: [0, 0, 1] → Nokta 1, instance 2'ye ait  
Nokta 2: [1, 0, 0] → Nokta 2, instance 0'a ait
```

### 2. **Predicted Masks (Model Çıktı Maskeleri)**
**Yer**: `mask_module()` fonksiyonu
**Kullanım**: Model her query için tüm noktalara bir "bu nokta bu instance'a ait mi?" skoru üretir

```386:400:svgnet/model/decoder.py
def mask_module(self, query_feat, mask_features,stage_list=None,step=None,ret_attn_mask=False):
    mask_embed = self.mask_embed_head(query_feat)  # Query'den mask embedding
    output_masks = mask_embed @ mask_features.T   # Her query × Her nokta skoru
```

**Şekil**: `(num_queries, N)`
- Her satır bir query'yi temsil eder
- Her sütun bir noktayı temsil eder
- Değer: Bu noktanın bu query'ye (instance'a) ait olma olasılığı (0-1 arası, sigmoid sonrası)

**İşlev**:
- Model `num_queries` adet potansiyel instance tespit eder
- Her query için tüm noktalara bir "membership score" verir
- Bu skorlar sigmoid ile [0,1] aralığına alınır

### 3. **Semantic Segmentation için Mask Kullanımı**
**Yer**: `semantic_inference()` fonksiyonu
**Kullanım**: Mask'ları kullanarak her nokta için semantic class tahmini yapılır

```131:136:svgnet/model/svgnet.py
def semantic_inference(self, mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[...,:-1] # Q,C - Her query için class prob
    mask_pred = mask_pred.sigmoid() # Q,G - Her query × Her nokta skoru
    semseg = torch.einsum("bqc,bqg->bgc", mask_cls, mask_pred)
    # Einstein sum: Her nokta için tüm query'lerin class prob × mask score toplamı
```

**Nasıl Çalışır**:
1. `mask_cls`: `(num_queries, num_classes)` - Her query'nin hangi class'ta olduğu olasılığı
2. `mask_pred`: `(num_queries, N)` - Her query'nin her noktaya ait olma skoru
3. `semseg`: `(N, num_classes)` - Her nokta için her class'ın toplam olasılığı

**Örnek**:
```
Query 1: class="kapı" (prob=0.9), nokta 5'e aitlik=0.8
Query 2: class="pencere" (prob=0.7), nokta 5'e aitlik=0.3
→ Nokta 5'in semantic score: 
   - kapı: 0.9 × 0.8 = 0.72
   - pencere: 0.7 × 0.3 = 0.21
   → Nokta 5 = "kapı" olarak tahmin edilir
```

### 4. **Instance Segmentation için Mask Kullanımı**
**Yer**: `instance_inference()` fonksiyonu
**Kullanım**: Mask'ları kullanarak hangi noktaların hangi instance'a ait olduğu belirlenir

```138:179:svgnet/model/svgnet.py
def instance_inference(self,mask_cls,mask_pred,overlap_threshold=0.8):
    # 1. Her query için class ve score hesapla
    scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
    
    # 2. Düşük score'lu ve background query'leri filtrele
    keep = labels.ne(self.num_classes) & (scores >= self.test_object_score)
    cur_masks = mask_pred[keep]  # Filtrelenmiş mask'lar
    
    # 3. Score × Mask = weighted masks
    cur_prob_masks = cur_scores[..., None] * cur_masks
    
    # 4. Her nokta için en yüksek skorlu query'yi bul (argmax)
    cur_mask_ids = cur_prob_masks.argmax(0)  # Her nokta hangi query'ye ait?
    
    # 5. Her query için final mask oluştur
    for k in range(cur_classes.shape[0]):
        mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)  # Binary mask
        # Overlap kontrolü vs.
        results.append({
            "masks": mask.cpu().numpy(),  # Binary mask: (N,) → [0,1,0,0,1,1,...]
            "labels": pred_class,
            "scores": pred_score
        })
```

**İşlem Adımları**:
1. **Filtreleme**: Düşük güvenilir query'leri çıkar (background, düşük score)
2. **Weighted Masks**: Score × Mask → Daha güvenilir query'ler daha ağırlıklı
3. **Argmax**: Her nokta için en yüksek skorlu query'yi seç
4. **Binary Mask Oluşturma**: 
   - Nokta o query'ye ait mi? (cur_mask_ids == k)
   - Query'nin o nokta için skoru yeterli mi? (cur_masks[k] >= 0.5)
   - İkisi de True ise → o nokta bu instance'a ait

**Örnek Output**:
```python
instances = [
    {
        "masks": np.array([0,0,0,1,1,1,0,0,...]),  # Nokta 3,4,5 bu instance'a ait
        "labels": 1,  # "kapı" class'ı
        "scores": 0.85
    },
    {
        "masks": np.array([1,1,0,0,0,0,1,1,...]),  # Nokta 0,1,6,7 bu instance'a ait
        "labels": 7,  # "pencere" class'ı
        "scores": 0.72
    }
]
```

### 5. **Attention Masks (Cross-Attention için)**
**Yer**: `mask_module()` içinde `ret_attn_mask=True` olduğunda
**Kullanım**: Cross-attention sırasında hangi noktalara bakılacağını belirler

```308:322:svgnet/model/decoder.py
output_class, outputs_mask,attn_mask = self.mask_module(queries, mask_features, 
                                                        stage_list, total_step-i,
                                                        ret_attn_mask=True)
# attn_mask: Hangi noktalara attention verilmeyeceğini belirler
output = self.cross_attention[decoder_counter][i](
    queries,
    src_pcd,
    memory_mask=attn_mask.transpose(0,1),  # Attention mask uygulanır
    ...
)
```

**İşlev**: 
- Query'ler point cloud feature'larına bakarken, hangi noktalara odaklanacağını belirler
- Mask ile belirlenmemiş noktalar ignore edilir (efficiency için)

---

## ÖZET: Mask Akışı

```
1. Ground Truth Masks (Training)
   → prepare_targets() → (N, K) binary masks
   
2. Model Prediction
   → mask_module() → (num_queries, N) probability masks
   
3. Semantic Segmentation
   → semantic_inference()
   → mask_cls × mask_pred → (N, num_classes)
   
4. Instance Segmentation  
   → instance_inference()
   → Filter → Argmax → Binary masks → (N,) per instance
   
5. Final Output
   → Her instance için:
      - masks: (N,) binary array [0,1,0,0,1,...]
      - labels: int (class ID)
      - scores: float (confidence)
```

**Neden Masks Önemli?**
- **Instance Segmentation**: Hangi noktaların hangi instance'a ait olduğunu belirler
- **Semantic Segmentation**: Mask'lar üzerinden nokta bazlı semantic class tahmini yapılır
- **Efficiency**: Attention mask ile gereksiz hesaplamalar önlenir
- **Training**: Model mask'ları öğrenerek instance'ları nasıl segment edeceğini öğrenir

---

## ÖNEMLİ NOTLAR

1. **Normalizasyon**: Koordinatlar JSON'dan okunurken [0,1] aralığına normalize edilir, sonra mean-center edilir.

2. **Padding**: Nokta sayısı minimum 2048'e çıkarılır (padding ile).

3. **Feature Kanalları**: 
   - 7 kanal: arc angle, normalized length, 4 channel command type, normalized width

4. **Downsampling**: Encoder'da 4× downsampling yapılır (her encoder layer'da).

5. **Queries**: Model `num_queries` adet öğrenilmiş query embedding kullanır (DETR tarzı).

6. **Layer Fusion**: Layer ID'lere göre feature fusion yapılır (decoder'da).

7. **Mask Threshold**: Instance inference'da `cur_masks[k] >= 0.5` threshold'u kullanılır - 0.5'ten büyük skorlar instance'a ait kabul edilir.

