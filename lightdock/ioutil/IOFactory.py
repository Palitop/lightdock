from lightdock.ioutil.IO import IO
from lightdock.ioutil.MMCIFIO import MMCIFIO
# rom .PDBIO import PDBIO
from lightdock.ioutil.IOExceptions import EmptyFileNameError, UnsupportedFileTypeError


class IOFactory:

    def get_instance(file_name: str) -> IO:
        if not file_name:
            raise EmptyFileNameError()

        file_type = IOFactory.get_file_type(file_name)

        if file_type in ["mmcif", "cif"]:
            return MMCIFIO()

        if file_type == "pdb":
            # return PDBIO()
            pass

        raise UnsupportedFileTypeError(file_type)

    def get_file_type(file_name: str) -> str:
        return file_name.split('.')[-1].lower()
