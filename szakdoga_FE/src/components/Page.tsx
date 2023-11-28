import { Box, Card, Typography } from "@mui/material";
import Divider from "@mui/material/Divider";
import { roundTo } from "round-to";
type Props = {
  title: string;
  children?: React.ReactNode;
  r2Score?: number;
};

const Page: React.FC<Props> = ({ title, children, r2Score }) => {
  return (
    <Card
      variant="outlined"
      style={{
        height: "90%",
      }}
    >
      <Box sx={{ paddingLeft: "2rem",display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="h4" component="h2">
          {title}
        </Typography>
        {r2Score && (
          <Typography variant="h6" component="h6">
            R2_Score: {roundTo(r2Score, 2)}
          </Typography>
        )}
      </Box>
      <Box sx={{ paddingBottom: "2rem" }}>
        <Divider />
      </Box>
      {children}
    </Card>
  );
};

export default Page;
