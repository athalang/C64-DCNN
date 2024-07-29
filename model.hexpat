#pragma endian little

enum matrixtypes : u8 {
    zero,
    csr,
};

bitfield matrixoptions {
  type : 1;
  padding : 3;
  ptr8 : 1;
  col8 : 1;
  row8 : 1;
  nnz8 : 1;
};

struct matrix {
    matrixoptions options;
    if (options.row8) u8 rows;
    else u16 rows;

    if (options.col8) u8 cols;
    else u16 cols;

    if (options.type == matrixtypes::csr) {
        if (options.nnz8) u8 nnz;
        else u16 nnz;

        if (options.ptr8) u8 *mrow[rows] : u16;
        else u16 *mrow[rows] : u16;

        if (options.col8) u8 *mcol[nnz] : u16;
        else u16 *mcol[nnz] : u16;

        u8 *mval[nnz] : u16;
    }
};

enum layertypes : u8 {
    relu,
    argmax,
    fullconn,
    conv,
    maxpool,
    flatten,
};

struct matrixptr {
    matrix *ptr: u16;
};

struct layer {
    layertypes type;

    if (type == layertypes::maxpool) {
        u8 kernel;
    }
    else if (type == layertypes::fullconn) {
        u16 ini;
        u16 outi;
        u16 *bias[outi] : u16;
        u16 *scale[outi] : u16;
        matrix *matrix : u16;
    }
    else if (type == layertypes::conv) {
        u16 ini;
        u16 outi;
        u8 kernel;
        u8 pad;
        u16 *scale[outi] : u16;
        matrixptr *matrix[ini * outi] : u16;
    }
};

struct layerptr {
    layer *ptr: u16;
};

u8 num_of_layers @ 0x00;
layerptr layers[num_of_layers] @ 0x01;