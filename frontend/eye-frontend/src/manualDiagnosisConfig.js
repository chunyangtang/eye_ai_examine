// Centralized manual diagnosis configuration (disease labels and common phrases)
export const MANUAL_DISEASE_ORDER = [
  '青光眼',
  '糖网',
  'AMD',
  '病理性近视',
  '豹纹状眼底',
  'RVO',
  'RAO',
  '视网膜脱离',
  '白内障',
  '正常',
];

export const MANUAL_DISEASE_INFO = {
  青光眼: {
    chinese: '青光眼',
    english: 'Glaucoma',
    fullName: '青光眼 (Glaucoma)',
    shortName: 'Glaucoma',
  },
  糖网: {
    chinese: '糖网',
    english: 'Diabetic Retinopathy',
    fullName: '糖网 (Diabetic Retinopathy)',
    shortName: 'DR',
  },
  AMD: {
    chinese: '年龄相关性黄斑变性',
    english: 'Age-related Macular Degeneration',
    fullName: '年龄相关性黄斑变性 (Age-related Macular Degeneration)',
    shortName: 'AMD',
  },
  病理性近视: {
    chinese: '病理性近视',
    english: 'Pathological Myopia',
    fullName: '病理性近视 (Pathological Myopia)',
    shortName: 'PM',
  },
  高度近视: {
    chinese: '高度近视',
    english: 'High Myopia',
    fullName: '高度近视 (High Myopia)',
    shortName: 'High Myopia',
  },
  豹纹状眼底: {
    chinese: '豹纹状眼底',
    english: 'Tessellated Fundus',
    fullName: '豹纹状眼底 (Tessellated Fundus)',
    shortName: 'Tessellated Fundus',
  },
  RVO: {
    chinese: '视网膜静脉阻塞',
    english: 'Retinal Vein Occlusion',
    fullName: '视网膜静脉阻塞 (RVO)',
    shortName: 'RVO',
  },
  RAO: {
    chinese: '视网膜动脉阻塞',
    english: 'Retinal Artery Occlusion',
    fullName: '视网膜动脉阻塞 (RAO)',
    shortName: 'RAO',
  },
  视网膜脱离: {
    chinese: '视网膜脱离',
    english: 'Retinal Detachment',
    fullName: '视网膜脱离 (Retinal Detachment)',
    shortName: 'RD',
  },
  其它视网膜病: {
    chinese: '其它视网膜病',
    english: 'Other Retinal Diseases',
    fullName: '其它视网膜病 (Other Retinal)',
    shortName: 'Other Retinal',
  },
  其它黄斑病变: {
    chinese: '其它黄斑病变',
    english: 'Other Macular Diseases',
    fullName: '其它黄斑病变 (Other Macular)',
    shortName: 'Other Macular',
  },
  白内障: {
    chinese: '白内障',
    english: 'Cataract',
    fullName: '白内障 (Cataract)',
    shortName: 'Cataract',
  },
  正常: {
    chinese: '正常',
    english: 'Normal',
    fullName: '正常 (Normal)',
    shortName: 'Normal',
  },
};

export const MANUAL_PHRASE_ALIASES = {
  糖网: 'DR',
  DR: 'DR',
  '视网膜动脉阻塞': '视网膜动脉阻塞',
  RAO: '视网膜动脉阻塞',
  '视网膜静脉阻塞': 'RVO',
  RVO: 'RVO',
  '病理性近视': '病理性近视',
  '高度近视': '病理性近视',
  '豹纹状眼底': '豹纹状眼底',
  青光眼: '青光眼',
  AMD: 'AMD',
  '视网膜脱离': '视网膜脱离',
};

export const MANUAL_COMMON_PHRASES = {
  separator: ', ',
  // keep misspelling for compatibility with requested schema
  seperator: ', ',
  common_phrases: {
    青光眼: [
      ['视盘色淡', '视盘线状出血'],
    ],
    DR: [
      ['微动脉瘤', '出血', '硬性渗出', '棉绒斑', '静脉串珠样改变', 'IRMA', '新生血管'],
      ['玻璃体出血', '纤维增殖膜', '牵拉性视网膜脱离', '视盘新生血管', '黄斑中心凹光反射弥散或消失'],
    ],
    AMD: [
      ['玻璃膜疣/Drusen', 'RPE色素改变', '出血', '黄斑瘢痕', '黄斑萎缩性改变'],
    ],
    RVO: [
      ['分支', '中央', '视网膜静脉迂曲扩张', '火焰状出血', '硬渗'],
      ['棉绒斑', '血管白线', '黄斑中心凹光反射弥散或消失'],
    ],
    视网膜动脉阻塞: [
      ['樱桃红斑', '视网膜苍白', '动脉变细'],
    ],
    视网膜脱离: [
      ['灰白色隆起', '视网膜裂孔', '视网膜皱褶', '无血管区'],
    ],
    病理性近视: [
      ['豹纹状眼底', '视盘倾斜', '后极萎缩斑', '色素改变', '漆裂纹', '脉络膜新生血管'],
    ],
    豹纹状眼底: [
      ['豹纹状眼底', '视盘倾斜'],
    ],
  },
};

export default {
  MANUAL_DISEASE_ORDER,
  MANUAL_DISEASE_INFO,
  MANUAL_COMMON_PHRASES,
  MANUAL_PHRASE_ALIASES,
};
