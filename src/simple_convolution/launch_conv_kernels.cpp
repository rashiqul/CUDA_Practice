void setup_for_1D_basic_conv(void);
void setup_for_1D_tiled_conv(void);
void setup_for_2D_basic_conv(void);
void setup_for_2D_conv_tiled(void);

int main(int argc, char *argv[])
{
    setup_for_1D_basic_conv();
    setup_for_1D_tiled_conv();
    setup_for_2D_basic_conv();
    setup_for_2D_conv_tiled();

    return 0;
}
