from lightdock.ioutil.IOFactory import IOFactory


class TestUtils:

    def compare_biological_content(file1, file2, tol=1e-6):
        def float_equal(a, b, tol=tol):
            return abs(a - b) <= tol

        io1 = IOFactory(file1).get_instance()
        atoms1, residues1, chains1 = io1.parse_complex_from_file(file1)

        io2 = IOFactory(file2).get_instance()
        atoms2, residues2, chains2 = io2.parse_complex_from_file(file2)

        if len(atoms1) != len(atoms2):
            return False
        if len(residues1) != len(residues2):
            return False
        if len(chains1) != len(chains2):
            return False

        for a1, a2 in zip(atoms1, atoms2):
            if a1.name != a2.name:
                return False
            if a1.alternative != a2.alternative:
                return False
            if a1.chain_id != a2.chain_id:
                return False
            if a1.residue_name != a2.residue_name:
                return False
            if a1.residue_number != a2.residue_number:
                return False
            if a1.residue_insertion != a2.residue_insertion:
                return False
            if not float_equal(a1.x, a2.x):
                return False
            if not float_equal(a1.y, a2.y):
                return False
            if not float_equal(a1.z, a2.z):
                return False
            if not float_equal(a1.occupancy, a2.occupancy):
                return False
            if not float_equal(a1.b_factor, a2.b_factor):
                return False
            if a1.element != a2.element:
                return False

        for r1, r2 in zip(residues1, residues2):
            if r1.name != r2.name:
                return False
            if r1.number != r2.number:
                return False
            if r1.insertion != r2.insertion:
                return False

        for c1, c2 in zip(chains1, chains2):
            if c1.cid != c2.cid:
                return False

        return True
