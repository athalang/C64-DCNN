#pragma endian little

enum matrixtypes : u8 {
    zero,
    csr,
};

struct matrix {
    u8 type;
    u16 rows;
    u16 cols;

    if (type == matrixtypes::csr) {
        u16 nnz;
        u16 *mrow[rows] : u16;
        u16 *mcol[nnz] : u16;
        u8 *mval[nnz] : u16;
    }
};

enum forwardtypes : u8 {
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

struct forward {
    forwardtypes type;

    if (type == forwardtypes::maxpool) {
        u8 kernel;
    }
    else if (type == forwardtypes::fullconn) {
        u16 ini;
        u16 outi;
        u16 *bias[outi] : u16;
        u16 *scale[outi] : u16;
        matrix *matrix : u16;
    }
    else if (type == forwardtypes::conv) {
        u16 ini;
        u16 outi;
        u8 kernel;
        u8 pad;
        u16 *scale[outi] : u16;
        matrixptr *matrix[ini * outi] : u16;
    }
};

struct forwardptr {
    forward *ptr: u16;
};

u8 num_of_forwards @ 0x00;
forwardptr forwards[num_of_forwards] @ 0x01;