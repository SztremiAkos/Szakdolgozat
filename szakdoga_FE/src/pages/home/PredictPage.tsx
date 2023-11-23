import {
  Autocomplete,
  Box,
  Button,
  Card,
  CardContent,
  TextField,
} from "@mui/material";
import systems from "../../configs/systemConfig";
import { RefObject, createRef, useEffect, useState } from "react";

type Props = {
  setSelectedSystem: (system: string) => void;
  onPredict: (params: number[]) => void;
  selectedSystem: string;
};
const PredictPage = (props: Props) => {
  const { setSelectedSystem, onPredict, selectedSystem } = props;

  const [inputFields, setInputFields] = useState<string[]>([]);
  const [inputRefs, setInputRefs] = useState<RefObject<any>[]>([]);

  useEffect(() => {
    const system = systems.filter((a) => a.name === selectedSystem)[0];
    if (system) {
      setInputFields(system.paramLabels);
    } else {
      setInputFields([]);
    }

    setInputRefs((elRefs: any) =>
      Array(inputFields.length)
        .fill(0)
        .map((_, i) => elRefs[i] || createRef())
    );
  }, [inputFields.length, selectedSystem]);
  return (
    <Card
      variant="outlined"
      style={{
        height: "90%",
        padding: "5rem",
      }}
    >
      <CardContent>
        <Box>
          <Box
            sx={{
              justifyContent: "center",
              gap: 2,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <Autocomplete
              disablePortal
              id="combo-box-demo"
              onChange={(_, systemName) => {
                setSelectedSystem(systemName ?? "");
              }}
              sx={{ width: 300 }}
              options={systems.map((system) => system.name)}
              renderInput={(params) => <TextField {...params} label="System" />}
            />
            {inputFields.map((label, index) => {
              return (
                <TextField
                  key={index}
                  type="number"
                  label={label}
                  inputRef={inputRefs[index]}
                  sx={{ width: 300 }}
                ></TextField>
              );
            })}
            {selectedSystem && (
              <>
                <Button
                  variant="outlined"
                  onClick={() =>
                    onPredict(
                      inputRefs.map((ref) => ref.current.value)
                    )
                  }
                >
                  Predict
                </Button>
              </>
            )}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default PredictPage;