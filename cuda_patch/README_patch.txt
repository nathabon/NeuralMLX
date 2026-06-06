Patch CUDA rapide
==================

Fichiers modifiés :
- backend_cupy.py : conv2d ne passe plus par cp.asnumpy -> torch CPU -> cp.array. Elle reste en CuPy/CUDA via im2col + matmul.
- neuralLayers.py : le backward des convolutions stridées utilise mx._im2col_strided, vectorisé sur kH*kW, au lieu de other.im2col_strided qui boucle sur H_out*W_out.

Tu peux remplacer tes fichiers originaux par ceux de ce dossier, puis relancer avec tes prints de timing sample/forward/backward.
